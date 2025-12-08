"""
Test the trained body-adaptive pose model on test dataset.
Calculates accuracy and performance metrics.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from collections import defaultdict
from pose_evaluator import BodyAdaptivePoseEvaluator

# 5 poses to test
POSE_CLASSES = [
    "Child_Pose_or_Balasana_",
    "Tree_Pose_or_Vrksasana_",
    "Warrior_I_Pose_or_Virabhadrasana_I_",
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_",
    "Plank_Pose_or_Kumbhakasana_"
]


def extract_landmarks(image_path):
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


def find_pose_folder(data_dir, pose_name):
    """Find the actual folder name for a pose"""
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


def test_model_on_dataset(evaluator, test_dir, pose_classes):
    """Test the model on test dataset"""
    print("=" * 70)
    print("🧪 Testing Model on Test Dataset")
    print("=" * 70)
    
    results = {
        'total_tested': 0,
        'total_correct': 0,
        'total_failed_extraction': 0,
        'per_pose': defaultdict(lambda: {'tested': 0, 'correct': 0, 'failed': 0}),
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'scores': []
    }
    
    # Test each pose
    for true_pose in pose_classes:
        pose_path = os.path.join(test_dir, true_pose)
        
        if not os.path.exists(pose_path):
            print(f"⚠️  {true_pose} not found in test dataset, skipping...")
            continue
        
        image_files = [f for f in os.listdir(pose_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📁 Testing {true_pose}")
        print(f"   Found {len(image_files)} test images")
        
        for img_file in image_files:
            img_path = os.path.join(pose_path, img_file)
            landmarks = extract_landmarks(img_path)
            
            if landmarks is None:
                results['total_failed_extraction'] += 1
                results['per_pose'][true_pose]['failed'] += 1
                continue
            
            # Test pose detection
            best_pose = None
            best_score = 0.0
            second_best_score = 0.0
            all_scores = {}
            
            for pose_name in pose_classes:
                if pose_name in evaluator.ideal_poses:
                    score, _ = evaluator.score_pose(landmarks, pose_name)
                    all_scores[pose_name] = score
                    if score > best_score:
                        second_best_score = best_score
                        best_score = score
                        best_pose = pose_name
                    elif score > second_best_score:
                        second_best_score = score
            
            # Prediction (optimized threshold with score gap check)
            min_score = 0.25
            score_gap = best_score - second_best_score if len(all_scores) > 1 else best_score
            
            if best_score > min_score and (score_gap > 0.05 or best_score > 0.6):
                predicted_pose = best_pose
            else:
                predicted_pose = None
            
            results['total_tested'] += 1
            results['per_pose'][true_pose]['tested'] += 1
            results['confusion_matrix'][true_pose][predicted_pose or 'None'] += 1
            results['scores'].append({
                'true_pose': true_pose,
                'predicted_pose': predicted_pose,
                'score': best_score,
                'all_scores': all_scores
            })
            
            if predicted_pose == true_pose:
                results['total_correct'] += 1
                results['per_pose'][true_pose]['correct'] += 1
    
    return results


def print_results(results):
    """Print test results"""
    print("\n" + "=" * 70)
    print("📊 Test Results")
    print("=" * 70)
    
    # Overall accuracy
    if results['total_tested'] > 0:
        accuracy = (results['total_correct'] / results['total_tested']) * 100
        print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")
        print(f"   Correct: {results['total_correct']}/{results['total_tested']}")
        print(f"   Failed to extract landmarks: {results['total_failed_extraction']}")
    else:
        print("\n❌ No images tested!")
        return
    
    # Per-pose accuracy
    print("\n📈 Per-Pose Accuracy:")
    print("-" * 70)
    for pose in sorted(results['per_pose'].keys()):
        stats = results['per_pose'][pose]
        if stats['tested'] > 0:
            pose_accuracy = (stats['correct'] / stats['tested']) * 100
            print(f"  {pose[:40]:<40} {pose_accuracy:6.2f}% ({stats['correct']}/{stats['tested']})")
    
    # Confusion matrix
    print("\n🔀 Confusion Matrix:")
    print("-" * 70)
    print(f"{'True\\Predicted':<30}", end="")
    for pose in POSE_CLASSES:
        print(f"{pose[:15]:<15}", end="")
    print("None")
    print("-" * 70)
    
    for true_pose in POSE_CLASSES:
        if true_pose in results['confusion_matrix']:
            print(f"{true_pose[:30]:<30}", end="")
            for pred_pose in POSE_CLASSES:
                count = results['confusion_matrix'][true_pose].get(pred_pose, 0)
                print(f"{count:<15}", end="")
            none_count = results['confusion_matrix'][true_pose].get('None', 0)
            print(f"{none_count}")
    
    # Score statistics
    if results['scores']:
        scores = [r['score'] for r in results['scores']]
        correct_scores = [r['score'] for r in results['scores'] if r['predicted_pose'] == r['true_pose']]
        incorrect_scores = [r['score'] for r in results['scores'] if r['predicted_pose'] != r['true_pose'] and r['predicted_pose'] is not None]
        
        print("\n📊 Score Statistics:")
        print("-" * 70)
        if scores:
            print(f"  Average Score (All): {np.mean(scores):.3f}")
        if correct_scores:
            print(f"  Average Score (Correct): {np.mean(correct_scores):.3f}")
        if incorrect_scores:
            print(f"  Average Score (Incorrect): {np.mean(incorrect_scores):.3f}")
    
    print("\n" + "=" * 70)


def main():
    """Main testing function"""
    # Load model
    model_path = "body_adaptive_pose_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Run: python train_body_adaptive.py")
        return
    
    print("Loading model...")
    evaluator = BodyAdaptivePoseEvaluator()
    evaluator.load(model_path)
    print(f"✅ Model loaded with {len(evaluator.ideal_poses)} poses\n")
    
    # Find actual pose folders in test dataset
    test_dir = "data_set/test"
    actual_pose_classes = []
    for pose_name in POSE_CLASSES:
        actual_folder = find_pose_folder(test_dir, pose_name)
        if actual_folder:
            actual_pose_classes.append(actual_folder)
            print(f"✅ Found test folder: {pose_name} → {actual_folder}")
        else:
            print(f"⚠️  Not found: {pose_name}")
    
    if len(actual_pose_classes) == 0:
        print("\n❌ No test pose folders found!")
        return
    
    # Test model
    results = test_model_on_dataset(evaluator, test_dir, actual_pose_classes)
    
    # Print results
    print_results(results)
    
    # Save results
    with open("test_results.txt", "w") as f:
        f.write("Test Results Summary\n")
        f.write("=" * 70 + "\n")
        if results['total_tested'] > 0:
            accuracy = (results['total_correct'] / results['total_tested']) * 100
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
            f.write(f"Correct: {results['total_correct']}/{results['total_tested']}\n")
            f.write(f"Failed extractions: {results['total_failed_extraction']}\n\n")
            
            f.write("Per-Pose Accuracy:\n")
            for pose in sorted(results['per_pose'].keys()):
                stats = results['per_pose'][pose]
                if stats['tested'] > 0:
                    pose_accuracy = (stats['correct'] / stats['tested']) * 100
                    f.write(f"  {pose}: {pose_accuracy:.2f}% ({stats['correct']}/{stats['tested']})\n")
    
    print("\n💾 Results saved to test_results.txt")


if __name__ == "__main__":
    main()

