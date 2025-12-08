"""
Body-Adaptive Yoga Pose Evaluator
Uses ML model to evaluate poses, adapting to different body types.
"""

import argparse
import os
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Try to load ML model
try:
    from pose_evaluator import BodyAdaptivePoseEvaluator
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False
    print("Warning: pose_evaluator not found. Using math-based evaluation.")


# Global ML model (loaded once)
ml_evaluator = None
pose_classes = []


def load_ml_model():
    """Load the trained ML model if available"""
    global ml_evaluator, pose_classes
    
    if not ML_MODEL_AVAILABLE:
        return False
    
    model_path = "body_adaptive_pose_model.pkl"
    classes_path = "pose_classes.txt"
    
    if not os.path.exists(model_path):
        print(f"⚠️  ML model not found at {model_path}")
        print("   Run: python train_body_adaptive.py")
        return False
    
    try:
        ml_evaluator = BodyAdaptivePoseEvaluator()
        ml_evaluator.load(model_path)
        
        # Load pose classes
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                pose_classes = [line.strip() for line in f.readlines()]
        else:
            # Fallback: use keys from model
            pose_classes = list(ml_evaluator.ideal_poses.keys())
        
        print(f"✅ Loaded ML model with {len(pose_classes)} poses:")
        for pose in pose_classes:
            print(f"   - {pose}")
        return True
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        return False


def detect_pose_class(landmarks) -> Optional[str]:
    """
    Detect which pose class the user is doing.
    Optimized: score against all poses, return best match with confidence threshold.
    """
    if ml_evaluator is None or len(pose_classes) == 0:
        return None
    
    best_pose = None
    best_score = 0.0
    second_best_score = 0.0
    
    scores = {}
    for pose_name in pose_classes:
        score, _ = ml_evaluator.score_pose(landmarks, pose_name)
        scores[pose_name] = score
        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_pose = pose_name
        elif score > second_best_score:
            second_best_score = score
    
    # Optimized threshold: require minimum score AND significant gap from second best
    min_score = 0.25  # Lower threshold for better detection
    score_gap = best_score - second_best_score
    
    # Return pose if:
    # 1. Score is above minimum threshold
    # 2. There's a clear winner (gap > 0.05) OR score is very high (>0.6)
    if best_score > min_score and (score_gap > 0.05 or best_score > 0.6):
        return best_pose
    return None


def evaluate_pose_ml(landmarks, pose_name: str) -> Tuple[float, Dict[str, bool]]:
    """Evaluate pose using ML model (body-adaptive)"""
    if ml_evaluator is None:
        return 0.0, {}
    
    score, feature_scores = ml_evaluator.score_pose(landmarks, pose_name)
    
    # Convert feature scores to cues (for compatibility)
    cues = {
        "feet_together": feature_scores.get('feet_ratio', 0) > 0.7,
        "weight_centered": feature_scores.get('shoulder_offset', 0) > 0.7,
        "arms_by_side": (feature_scores.get('left_arm_offset', 0) > 0.7 and 
                        feature_scores.get('right_arm_offset', 0) > 0.7),
        "torso_stack": feature_scores.get('hip_offset', 0) > 0.7,
        "head_aligned": feature_scores.get('head_offset', 0) > 0.7,
    }
    
    return score, cues


def evaluate_tadasana(
    landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
) -> Tuple[float, Dict[str, bool]]:
    """Fallback: Math-based evaluation (if ML not available)"""
    if not landmarks:
        return 0.0, {}

    lm = landmarks.landmark
    P = mp.solutions.pose.PoseLandmark

    needed = [
        P.LEFT_ANKLE,
        P.RIGHT_ANKLE,
        P.LEFT_HIP,
        P.RIGHT_HIP,
        P.LEFT_SHOULDER,
        P.RIGHT_SHOULDER,
        P.LEFT_WRIST,
        P.RIGHT_WRIST,
        P.NOSE,
    ]
    for idx in needed:
        if lm[idx].visibility < 0.5:
            return 0.0, {}

    left_ankle = lm[P.LEFT_ANKLE]
    right_ankle = lm[P.RIGHT_ANKLE]
    left_hip = lm[P.LEFT_HIP]
    right_hip = lm[P.RIGHT_HIP]
    left_sh = lm[P.LEFT_SHOULDER]
    right_sh = lm[P.RIGHT_SHOULDER]
    left_wrist = lm[P.LEFT_WRIST]
    right_wrist = lm[P.RIGHT_WRIST]
    nose = lm[P.NOSE]

    # Feet together (ankles close in X)
    feet_distance = abs(left_ankle.x - right_ankle.x)
    feet_ok = feet_distance <= 0.05
    feet_score = max(0.0, 1.0 - feet_distance / 0.10)

    # Torso stack & weight centered
    ankle_center_x = (left_ankle.x + right_ankle.x) / 2.0
    shoulder_center_x = (left_sh.x + right_sh.x) / 2.0
    hip_center_x = (left_hip.x + right_hip.x) / 2.0
    shoulder_offset = abs(shoulder_center_x - ankle_center_x)
    hip_offset = abs(hip_center_x - ankle_center_x)
    max_offset = 0.08
    torso_ok = shoulder_offset <= 0.06 and hip_offset <= 0.06
    torso_score = max(0.0, 1.0 - max(shoulder_offset, hip_offset) / max_offset)

    # Arms by side
    hip_y = (left_hip.y + right_hip.y) / 2.0
    left_arm_dx = abs(left_wrist.x - left_sh.x)
    right_arm_dx = abs(right_wrist.x - right_sh.x)
    vertical_ok = (
        left_sh.y + 0.02 < left_wrist.y < hip_y + 0.05
        and right_sh.y + 0.02 < right_wrist.y < hip_y + 0.05
    )
    arms_side_ok = (
        left_arm_dx <= 0.05
        and right_arm_dx <= 0.05
        and vertical_ok
    )
    arms_side_score = (
        max(0.0, 1.0 - max(left_arm_dx, right_arm_dx) / 0.15) if arms_side_ok else 0.0
    )

    # Head alignment
    nose_offset = abs(nose.x - shoulder_center_x)
    head_vec = np.array([nose.x - shoulder_center_x, nose.y - (left_sh.y + right_sh.y) / 2.0])
    vertical = np.array([0.0, -1.0])
    head_cos = np.dot(head_vec, vertical) / (np.linalg.norm(head_vec) * np.linalg.norm(vertical) + 1e-6)
    head_cos = float(np.clip(head_cos, -1.0, 1.0))
    head_angle = float(np.degrees(np.arccos(head_cos)))
    head_ok = nose_offset <= 0.05 and head_angle <= 15.0
    head_score = max(0.0, 1.0 - nose_offset / 0.08) * max(0.0, 1.0 - head_angle / 30.0)

    # Aggregate
    scores = [feet_score, torso_score, arms_side_score, head_score]
    overall = float(sum(scores) / len(scores))
    cues = {
        "feet_together": feet_ok,
        "weight_centered": torso_ok,
        "arms_by_side": arms_side_ok,
        "torso_stack": torso_ok,
        "head_aligned": head_ok,
    }
    return overall, cues


def format_pose_name(pose_name: str) -> str:
    """Format pose name for display"""
    # Clean up folder names for display
    name = pose_name.replace("_", " ")
    name = name.replace("or", "|")
    # Take first part before "|" or "_"
    if "|" in name:
        name = name.split("|")[0].strip()
    return name.title()


def run(video_source) -> None:
    mp_pose = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Determine if video_source is a file path or camera index
    is_video_file = isinstance(video_source, str) and os.path.exists(video_source)
    
    if is_video_file:
        cap = cv2.VideoCapture(video_source)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 Video file: {video_source}")
        print(f"   FPS: {video_fps}, Total frames: {total_frames}")
    else:
        cap = cv2.VideoCapture(int(video_source))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {video_source}")

    using_ml = ml_evaluator is not None
    mode_text = "ML (Body-Adaptive)" if using_ml else "Math-Based"
    
    print(f"🧘 Yoga Pose Evaluator - {mode_text}")
    if is_video_file:
        print("Press 'q' to quit, 'r' to replay video.")
    else:
        print("Press 'q' to quit.")
    
    frame_count = 0
    failed_reads = 0
    max_failed_reads = 10  # Allow some failed reads before giving up
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                failed_reads += 1
                if is_video_file:
                    print("Video ended. Press 'r' to replay or 'q' to quit.")
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('r'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
                        frame_count = 0
                        failed_reads = 0
                        continue
                    elif key == ord('q'):
                        break
                else:
                    if failed_reads >= max_failed_reads:
                        print(f"Failed to read frame from camera after {max_failed_reads} attempts. Exiting.")
                        print("💡 Tip: Make sure no other app is using the camera.")
                        break
                    # Continue trying for a few more frames
                    continue
            else:
                failed_reads = 0  # Reset counter on successful read
            
            frame_count += 1
            
            # Display frame number for video files
            if is_video_file:
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}/{total_frames}",
                    (40, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)

            frame.flags.writeable = True
            detected_pose = None
            score = 0.0
            cues = {}
            
            if results.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=(
                        drawing_styles.get_default_pose_landmarks_style()
                    ),
                )
                
                # Detect which pose (if ML available)
                if using_ml:
                    detected_pose = detect_pose_class(results.pose_landmarks)
                    
                    if detected_pose:
                        # Evaluate using ML
                        score, cues = evaluate_pose_ml(results.pose_landmarks, detected_pose)
                    else:
                        # No pose detected
                        score = 0.0
                else:
                    # Fallback to math-based
                    score, cues = evaluate_tadasana(results.pose_landmarks)
                    detected_pose = "Tadasana" if score > 0.5 else None

            # Display pose name
            y_offset = 40
            if detected_pose:
                pose_display = format_pose_name(detected_pose)
                cv2.putText(
                    frame,
                    f"Pose: {pose_display}",
                    (40, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                y_offset += 35
            else:
                cv2.putText(
                    frame,
                    "No pose detected",
                    (40, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (128, 128, 128),
                    2,
                )
                y_offset += 35

            # Display score
            cv2.putText(
                frame,
                f"Score: {score*100:5.1f}%",
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            y_offset += 30

            # Success message
            if detected_pose and score >= 0.8:
                cv2.putText(
                    frame,
                    "✓ Excellent!",
                    (40, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            elif detected_pose and score >= 0.6:
                cv2.putText(
                    frame,
                    "Good - Keep adjusting",
                    (40, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 200),
                    2,
                )
            elif detected_pose:
                cv2.putText(
                    frame,
                    "Adjust your pose",
                    (40, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 100, 255),
                    2,
                )

            # Mode indicator
            mode_color = (0, 255, 0) if using_ml else (255, 165, 0)
            cv2.putText(
                frame,
                f"Mode: {mode_text}",
                (40, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                mode_color,
                1,
            )

            cv2.imshow("Yoga Pose Evaluator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()
        print("Demo stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Body-Adaptive Yoga Pose Evaluator")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g., '0' for webcam) or video file path (e.g., 'video.mp4').",
    )
    args = parser.parse_args()
    
    # Convert to int if it's a number (camera index), otherwise keep as string (file path)
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source  # It's a file path
    
    # Load ML model
    load_ml_model()
    
    # Run
    try:
        run(video_source=video_source)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
