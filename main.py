"""
Minimal MediaPipe Pose viewer.
Opens the default webcam, runs MediaPipe Pose, and overlays landmarks.
"""

import argparse
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


def evaluate_tadasana(
    landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
) -> Tuple[float, Dict[str, bool]]:
    """Compute a simple performance score (0–1) for Tadasana and per-cue flags.

    Cues (mapped to your description):
    - feet_together         → Stand straight with feet together.
    - weight_centered       → Keep your weight equal on both feet.
    - arms_by_side          → Arms straight beside the body.
    - torso_stack           → Spine tall and straight (shoulders/hips over feet).
    - head_aligned          → Head straight, chin parallel to the floor.
    """
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
    feet_ok = feet_distance <= 0.05  # quite close, but allow tiny gap
    feet_score = max(0.0, 1.0 - feet_distance / 0.10)

    # Torso stack & weight centered: centers over feet
    ankle_center_x = (left_ankle.x + right_ankle.x) / 2.0
    shoulder_center_x = (left_sh.x + right_sh.x) / 2.0
    hip_center_x = (left_hip.x + right_hip.x) / 2.0
    shoulder_offset = abs(shoulder_center_x - ankle_center_x)
    hip_offset = abs(hip_center_x - ankle_center_x)
    max_offset = 0.08
    torso_ok = shoulder_offset <= 0.06 and hip_offset <= 0.06
    torso_score = max(0.0, 1.0 - max(shoulder_offset, hip_offset) / max_offset)

    # Arms by side: wrists roughly below shoulders and near hip in Y, and close in X
    hip_y = (left_hip.y + right_hip.y) / 2.0
    left_arm_dx = abs(left_wrist.x - left_sh.x)
    right_arm_dx = abs(right_wrist.x - right_sh.x)
    # Keep a small band so hands are not too close to shoulders or too low
    vertical_ok = (
        left_sh.y + 0.02 < left_wrist.y < hip_y + 0.05
        and right_sh.y + 0.02 < right_wrist.y < hip_y + 0.05
    )
    # Require wrists to be very close to the side of the torso
    arms_side_ok = (
        left_arm_dx <= 0.05
        and right_arm_dx <= 0.05
        and vertical_ok
    )
    # If the wrists are not in the side/hip band (e.g., arms lifted up),
    # drop the arm score heavily so overall score goes down.
    arms_side_score = (
        max(0.0, 1.0 - max(left_arm_dx, right_arm_dx) / 0.15) if arms_side_ok else 0.0
    )

    # Head alignment: nose roughly above shoulder center horizontally
    nose_offset = abs(nose.x - shoulder_center_x)
    # Also check that the line shoulder_center -> nose is close to vertical
    head_vec = np.array([nose.x - shoulder_center_x, nose.y - (left_sh.y + right_sh.y) / 2.0])
    vertical = np.array([0.0, -1.0])
    head_cos = np.dot(head_vec, vertical) / (np.linalg.norm(head_vec) * np.linalg.norm(vertical) + 1e-6)
    head_cos = float(np.clip(head_cos, -1.0, 1.0))
    head_angle = float(np.degrees(np.arccos(head_cos)))  # 0° means perfectly vertical
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


def is_tadasana_pose(
    landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
) -> bool:
    """Wrapper using the performance score with a threshold."""
    score, cues = evaluate_tadasana(landmarks)
    return (
        score >= 0.8
        and cues.get("feet_together", False)
        and cues.get("arms_by_side", False)
        and cues.get("torso_stack", False)
        and cues.get("head_aligned", False)
    )


def run(video_source: int = 0) -> None:
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

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {video_source}")

    print("MediaPipe Pose demo running. Press 'q' to quit.")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera. Exiting.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)

            frame.flags.writeable = True
            success_pose = False
            score = 0.0
            if results.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=(
                        drawing_styles.get_default_pose_landmarks_style()
                    ),
                )
                score, cues = evaluate_tadasana(results.pose_landmarks)
                # Success only when feet are together AND arms really by the side,
                # plus a reasonably high overall score.
                success_pose = (
                    score >= 0.8
                    and cues.get("feet_together", False)
                    and cues.get("arms_by_side", False)
                )

            if success_pose:
                cv2.putText(
                    frame,
                    "Success! Tadasana aligned",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 0),
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    "Stand tall: feet together, arms by side",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # Draw performance score
            cv2.putText(
                frame,
                f"Score: {score*100:5.1f}%",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("MediaPipe Pose Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()
        print("Demo stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MediaPipe Pose viewer.")
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Video source index (default: 0 for built-in webcam).",
    )
    args = parser.parse_args()
    run(video_source=args.source)


if __name__ == "__main__":
    main()

