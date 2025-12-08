"""
Train a body-adaptive pose evaluation model.
Learns ideal pose patterns from dataset, normalized by body proportions.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import joblib

# 5 simple yoga poses to train (confirmed in dataset)
POSE_CLASSES = [
    "Child_Pose_or_Balasana_",           # Child's Pose - seated, simple
    "Tree_Pose_or_Vrksasana_",           # Tree Pose - standing balance
    "Warrior_I_Pose_or_Virabhadrasana_I_",  # Warrior I - standing lunge
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_",  # Downward Dog
    "Plank_Pose_or_Kumbhakasana_"        # Plank - core pose
]


class BodyAdaptivePoseEvaluator:
    def __init__(self):
        self.ideal_poses = {}  # Ideal patterns per pose
        self.body_scalers = {}  # Normalize by body size
        
    def extract_landmarks(self, image_path):
        """Extract pose landmarks from an image"""
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        pose.close()
        
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None
    
    def normalize_by_body_size(self, landmarks):
        """
        Normalize landmarks by body size to handle different body types.
        Uses torso length as reference (hip to shoulder).
        """
        P = mp.solutions.pose.PoseLandmark
        
        # Get key points
        left_shoulder = landmarks[P.LEFT_SHOULDER]
        right_shoulder = landmarks[P.RIGHT_SHOULDER]
        left_hip = landmarks[P.LEFT_HIP]
        right_hip = landmarks[P.RIGHT_HIP]
        
        # Calculate body size reference (torso length)
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        
        # Torso length as normalization factor
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        if torso_length < 1e-6:  # Avoid division by zero
            return None
        
        # Normalize all landmarks relative to torso length
        normalized = []
        for lm in landmarks:
            # Center around hip (body center)
            x_norm = (lm.x - hip_center[0]) / torso_length
            y_norm = (lm.y - hip_center[1]) / torso_length
            z_norm = lm.z / torso_length  # Z is depth, normalize similarly
            normalized.extend([x_norm, y_norm, z_norm, lm.visibility])
        
        return np.array(normalized)
    
    def extract_relative_features(self, landmarks):
        """
        Extract relative features that adapt to body type.
        Uses ratios and angles instead of absolute distances.
        """
        P = mp.solutions.pose.PoseLandmark
        
        # Get key points
        left_ankle = landmarks[P.LEFT_ANKLE]
        right_ankle = landmarks[P.RIGHT_ANKLE]
        left_hip = landmarks[P.LEFT_HIP]
        right_hip = landmarks[P.RIGHT_HIP]
        left_shoulder = landmarks[P.LEFT_SHOULDER]
        right_shoulder = landmarks[P.RIGHT_SHOULDER]
        left_wrist = landmarks[P.LEFT_WRIST]
        right_wrist = landmarks[P.RIGHT_WRIST]
        nose = landmarks[P.NOSE]
        
        # Calculate body size reference
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        ankle_center = np.array([
            (left_ankle.x + right_ankle.x) / 2,
            (left_ankle.y + right_ankle.y) / 2
        ])
        
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        leg_length = np.linalg.norm(hip_center - ankle_center)
        
        if torso_length < 1e-6:
            return None
        
        features = []
        
        # 1. Feet distance relative to body width
        feet_distance = abs(left_ankle.x - right_ankle.x)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        feet_ratio = feet_distance / (shoulder_width + 1e-6)
        features.append(feet_ratio)
        
        # 2. Alignment ratios (relative to body size)
        shoulder_offset = abs(shoulder_center[0] - ankle_center[0]) / torso_length
        hip_offset = abs(hip_center[0] - ankle_center[0]) / torso_length
        features.extend([shoulder_offset, hip_offset])
        
        # 3. Arm position relative to torso
        left_arm_offset = abs(left_wrist.x - left_shoulder.x) / torso_length
        right_arm_offset = abs(right_wrist.x - right_shoulder.x) / torso_length
        features.extend([left_arm_offset, right_arm_offset])
        
        # 4. Arm vertical position (relative to body height)
        body_height = abs(shoulder_center[1] - ankle_center[1])
        left_wrist_y_ratio = (left_wrist.y - shoulder_center[1]) / (body_height + 1e-6)
        right_wrist_y_ratio = (right_wrist.y - shoulder_center[1]) / (body_height + 1e-6)
        features.extend([left_wrist_y_ratio, right_wrist_y_ratio])
        
        # 5. Head alignment (relative)
        head_offset = abs(nose.x - shoulder_center[0]) / torso_length
        features.append(head_offset)
        
        # 6. Body proportions
        torso_leg_ratio = torso_length / (leg_length + 1e-6)
        features.append(torso_leg_ratio)
        
        # 7. Add normalized landmark positions
        normalized_landmarks = self.normalize_by_body_size(landmarks)
        if normalized_landmarks is not None:
            features.extend(normalized_landmarks.tolist())
        
        return np.array(features)
    
    def learn_ideal_poses(self, data_dir, pose_classes):
        """Learn ideal pose patterns from dataset, normalized by body type"""
        print("Learning body-adaptive ideal poses...")
        print("=" * 60)
        
        for pose_name in pose_classes:
            pose_path = os.path.join(data_dir, pose_name)
            
            if not os.path.exists(pose_path):
                print(f"⚠️  Warning: {pose_path} not found, skipping...")
                continue
            
            all_features = []
            image_files = [f for f in os.listdir(pose_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\n📁 Processing {pose_name}")
            print(f"   Found {len(image_files)} images")
            
            processed = 0
            failed = 0
            
            for img_file in image_files:
                img_path = os.path.join(pose_path, img_file)
                landmarks = self.extract_landmarks(img_path)
                
                if landmarks:
                    features = self.extract_relative_features(landmarks)
                    if features is not None:
                        all_features.append(features)
                        processed += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            
            if len(all_features) > 0:
                all_features = np.array(all_features)
                
                # Learn ideal pattern (mean of normalized features)
                ideal_features = np.mean(all_features, axis=0)
                self.ideal_poses[pose_name] = ideal_features
                
                # Learn feature scaler (for different feature scales)
                scaler = StandardScaler()
                scaler.fit(all_features)
                self.body_scalers[pose_name] = scaler
                
                print(f"   ✅ Successfully processed: {processed} images")
                print(f"   ❌ Failed: {failed} images")
                print(f"   📊 Learned ideal pattern with {len(ideal_features)} features")
            else:
                print(f"   ❌ No valid features extracted for {pose_name}")
        
        print("\n" + "=" * 60)
        print(f"✅ Learned {len(self.ideal_poses)} body-adaptive poses")
        return len(self.ideal_poses)
    
    def score_pose(self, current_landmarks, pose_name):
        """
        Score current pose against ideal, adapting to body type.
        Returns: score (0-1), detailed feedback dict
        """
        if pose_name not in self.ideal_poses:
            return 0.0, {}
        
        # Extract relative features (normalized by body size)
        current_features = self.extract_relative_features(current_landmarks)
        if current_features is None:
            return 0.0, {}
        
        ideal_features = self.ideal_poses[pose_name]
        scaler = self.body_scalers[pose_name]
        
        # Normalize features
        current_norm = scaler.transform([current_features])[0]
        ideal_norm = scaler.transform([ideal_features])[0]
        
        # Calculate similarity (cosine similarity works well for normalized features)
        cosine_sim = 1 - cosine(current_norm, ideal_norm)
        cosine_sim = max(0.0, min(1.0, cosine_sim))  # Clamp to [0, 1]
        
        # Also calculate per-feature scores for detailed feedback
        feature_scores = {}
        feature_names = [
            'feet_ratio', 'shoulder_offset', 'hip_offset',
            'left_arm_offset', 'right_arm_offset',
            'left_wrist_y', 'right_wrist_y', 'head_offset'
        ]
        
        # Compare key features
        for i, name in enumerate(feature_names):
            if i < len(current_norm):
                diff = abs(current_norm[i] - ideal_norm[i])
                feature_scores[name] = max(0.0, 1.0 - diff)
        
        return cosine_sim, feature_scores
    
    def save(self, filepath):
        """Save learned models"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'ideal_poses': self.ideal_poses,
                'body_scalers': self.body_scalers
            }, f)
        print(f"\n💾 Saved body-adaptive model to {filepath}")
    
    def load(self, filepath):
        """Load learned models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.ideal_poses = data['ideal_poses']
            self.body_scalers = data['body_scalers']
        print(f"✅ Loaded body-adaptive model from {filepath}")


def find_pose_folder(data_dir, pose_name):
    """Find the actual folder name for a pose (handles variations)"""
    if os.path.exists(os.path.join(data_dir, pose_name)):
        return pose_name
    
    # Try variations
    variations = [
        pose_name.replace("_", " "),
        pose_name.replace("-", "_"),
        pose_name.replace("_", "-"),
    ]
    
    for var in variations:
        if os.path.exists(os.path.join(data_dir, var)):
            return var
    
    # Search for partial matches
    all_folders = os.listdir(data_dir)
    for folder in all_folders:
        if pose_name.lower() in folder.lower() or folder.lower() in pose_name.lower():
            return folder
    
    return None


def main():
    """Main training function"""
    print("=" * 60)
    print("🧘 Body-Adaptive Yoga Pose Trainer")
    print("=" * 60)
    
    data_dir = "data_set/train"
    
    # Find actual folder names
    print("\n🔍 Finding pose folders...")
    actual_pose_classes = []
    for pose_name in POSE_CLASSES:
        actual_folder = find_pose_folder(data_dir, pose_name)
        if actual_folder:
            actual_pose_classes.append(actual_folder)
            print(f"   ✅ {pose_name} → {actual_folder}")
        else:
            print(f"   ❌ {pose_name} → NOT FOUND")
    
    if len(actual_pose_classes) == 0:
        print("\n❌ No pose folders found! Please check your dataset.")
        return
    
    # Create evaluator and train
    evaluator = BodyAdaptivePoseEvaluator()
    num_poses = evaluator.learn_ideal_poses(data_dir, actual_pose_classes)
    
    if num_poses > 0:
        # Save model
        model_path = "body_adaptive_pose_model.pkl"
        evaluator.save(model_path)
        
        # Save pose class mapping
        with open("pose_classes.txt", "w") as f:
            for pose in actual_pose_classes:
                f.write(f"{pose}\n")
        print(f"💾 Saved pose classes to pose_classes.txt")
        
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print(f"   Trained on {num_poses} poses")
        print(f"   Model saved to: {model_path}")
        print("\n📝 Next steps:")
        print("   1. Run: python main.py")
        print("   2. The model will automatically load and evaluate poses")
        print("=" * 60)
    else:
        print("\n❌ Training failed - no poses were learned.")


if __name__ == "__main__":
    main()

