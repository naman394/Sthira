"""
Body-adaptive pose evaluator module.
Can be imported by main.py
"""

import numpy as np
import pickle
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import mediapipe as mp


class BodyAdaptivePoseEvaluator:
    """Body-adaptive pose evaluation using learned ideal patterns"""
    
    def __init__(self):
        self.ideal_poses = {}
        self.body_scalers = {}
    
    def normalize_by_body_size(self, landmarks):
        """Normalize landmarks by body size"""
        P = mp.solutions.pose.PoseLandmark
        
        left_shoulder = landmarks[P.LEFT_SHOULDER]
        right_shoulder = landmarks[P.RIGHT_SHOULDER]
        left_hip = landmarks[P.LEFT_HIP]
        right_hip = landmarks[P.RIGHT_HIP]
        
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        if torso_length < 1e-6:
            return None
        
        normalized = []
        for lm in landmarks:
            x_norm = (lm.x - hip_center[0]) / torso_length
            y_norm = (lm.y - hip_center[1]) / torso_length
            z_norm = lm.z / torso_length
            normalized.extend([x_norm, y_norm, z_norm, lm.visibility])
        
        return np.array(normalized)
    
    def extract_relative_features(self, landmarks):
        """Extract relative features that adapt to body type"""
        P = mp.solutions.pose.PoseLandmark
        
        left_ankle = landmarks[P.LEFT_ANKLE]
        right_ankle = landmarks[P.RIGHT_ANKLE]
        left_hip = landmarks[P.LEFT_HIP]
        right_hip = landmarks[P.RIGHT_HIP]
        left_shoulder = landmarks[P.LEFT_SHOULDER]
        right_shoulder = landmarks[P.RIGHT_SHOULDER]
        left_wrist = landmarks[P.LEFT_WRIST]
        right_wrist = landmarks[P.RIGHT_WRIST]
        nose = landmarks[P.NOSE]
        
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
        
        # Relative features
        feet_distance = abs(left_ankle.x - right_ankle.x)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        feet_ratio = feet_distance / (shoulder_width + 1e-6)
        features.append(feet_ratio)
        
        shoulder_offset = abs(shoulder_center[0] - ankle_center[0]) / torso_length
        hip_offset = abs(hip_center[0] - ankle_center[0]) / torso_length
        features.extend([shoulder_offset, hip_offset])
        
        left_arm_offset = abs(left_wrist.x - left_shoulder.x) / torso_length
        right_arm_offset = abs(right_wrist.x - right_shoulder.x) / torso_length
        features.extend([left_arm_offset, right_arm_offset])
        
        body_height = abs(shoulder_center[1] - ankle_center[1])
        left_wrist_y_ratio = (left_wrist.y - shoulder_center[1]) / (body_height + 1e-6)
        right_wrist_y_ratio = (right_wrist.y - shoulder_center[1]) / (body_height + 1e-6)
        features.extend([left_wrist_y_ratio, right_wrist_y_ratio])
        
        head_offset = abs(nose.x - shoulder_center[0]) / torso_length
        features.append(head_offset)
        
        torso_leg_ratio = torso_length / (leg_length + 1e-6)
        features.append(torso_leg_ratio)
        
        # Add normalized landmarks
        normalized_landmarks = self.normalize_by_body_size(landmarks)
        if normalized_landmarks is not None:
            features.extend(normalized_landmarks.tolist())
        
        return np.array(features)
    
    def score_pose(self, current_landmarks, pose_name):
        """Score current pose against ideal - OPTIMIZED VERSION"""
        if pose_name not in self.ideal_poses:
            return 0.0, {}
        
        current_features = self.extract_relative_features(current_landmarks)
        if current_features is None:
            return 0.0, {}
        
        ideal_features = self.ideal_poses[pose_name]
        scaler = self.body_scalers[pose_name]
        
        # Transform current features to normalized space
        current_norm = scaler.transform([current_features])[0]
        
        # In normalized space, ideal (mean of training) is approximately zero
        # But we'll compute it properly for better accuracy
        ideal_norm = scaler.transform([ideal_features])[0]
        
        # OPTIMIZED SCORING: Use multiple metrics for better accuracy
        
        # 1. Euclidean distance in normalized space (primary metric)
        euclidean_dist = np.linalg.norm(current_norm - ideal_norm)
        
        # OPTIMIZED: Use adaptive scaling based on feature dimension
        # For 141 features with std=1, typical distances:
        # - Very similar: 0-5
        # - Similar: 5-10
        # - Different: 10-20
        # - Very different: 20+
        # Use exponential decay with better scaling for higher scores
        scale_factor = 8.0  # Increased from 4.0 for better score distribution
        dist_score = np.exp(-euclidean_dist / scale_factor)
        
        # Boost scores for very close matches
        if euclidean_dist < 5.0:
            dist_score = min(1.0, dist_score * 1.2)  # Boost close matches
        
        # 2. Cosine similarity (if vectors are non-zero)
        cosine_score = 0.0
        current_norm_norm = np.linalg.norm(current_norm)
        ideal_norm_norm = np.linalg.norm(ideal_norm)
        
        if current_norm_norm > 1e-6 and ideal_norm_norm > 1e-6:
            cosine_sim = np.dot(current_norm, ideal_norm) / (current_norm_norm * ideal_norm_norm)
            cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))
            cosine_score = (cosine_sim + 1.0) / 2.0  # Convert to 0-1
        
        # 3. Feature-wise comparison (weighted by importance)
        # Key features get more weight - these are most discriminative
        key_feature_indices = [0, 1, 2, 3, 4, 7]  # feet_ratio, shoulder_offset, hip_offset, arm offsets, head_offset
        feature_scores_list = []
        
        for i in key_feature_indices:
            if i < len(current_norm):
                diff = abs(current_norm[i] - ideal_norm[i])
                # Normalized features: diff typically 0-2 for similar, 2-5 for different
                # Use gentler decay for better scores
                feat_score = np.exp(-diff / 2.0)  # Increased from 1.5 for better scores
                feature_scores_list.append(feat_score)
        
        feature_score = np.mean(feature_scores_list) if feature_scores_list else 0.0
        
        # Boost feature score if most key features match well
        if feature_score > 0.7:
            feature_score = min(1.0, feature_score * 1.1)
        
        # 4. Combined score with optimized weights
        # Give balanced weights for better score distribution
        if cosine_score > 0:
            similarity = (dist_score * 0.45 + cosine_score * 0.25 + feature_score * 0.30)
        else:
            similarity = (dist_score * 0.55 + feature_score * 0.45)
        
        # Final score boost for high-confidence matches
        if similarity > 0.5:
            # Apply gentle boost to high scores for better distribution
            similarity = min(1.0, similarity * 1.05)
        
        # Ensure score is in [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        # Calculate per-feature scores for detailed feedback
        feature_scores = {}
        feature_names = [
            'feet_ratio', 'shoulder_offset', 'hip_offset',
            'left_arm_offset', 'right_arm_offset',
            'left_wrist_y', 'right_wrist_y', 'head_offset'
        ]
        
        for i, name in enumerate(feature_names):
            if i < len(current_norm):
                diff = abs(current_norm[i] - ideal_norm[i])
                # Use exponential decay with better scaling for higher scores
                feature_scores[name] = max(0.0, min(1.0, np.exp(-diff / 2.0)))
        
        return similarity, feature_scores
    
    def load(self, filepath):
        """Load learned models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.ideal_poses = data['ideal_poses']
            self.body_scalers = data['body_scalers']

