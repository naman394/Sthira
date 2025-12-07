# Tadasana Pose Check – MediaPipe + OpenCV Demo

This project is a minimal real‑time pose feedback demo built on top of [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose) and OpenCV.  
Right now it focuses on a **single pose: Tadasana (Mountain Pose)** and gives you a live **performance score** plus a “success” indicator when your alignment is good enough.

All earlier complex yoga trainer / personalization logic has been removed so we can iterate from a clean, understandable base.

---

## What the demo does

When you run `main.py` with a webcam connected:

- It uses **MediaPipe Pose** to track 33 body landmarks in real time.
- It overlays the skeleton on top of your webcam feed.
- On every frame it computes how close you are to a simple definition of **Tadasana**:
  - **Feet together** – ankles almost touching (legs straight and parallel).
  - **Weight centered** – hips and shoulders stacked above your feet (not leaning).
  - **Spine tall** – shoulders over hips.
  - **Arms straight by your sides** – wrists close to the body, hanging roughly between hips and shoulders (not forward, not out, not lifted).
  - **Head straight** – nose centered above your chest, head not tilted more than ~15°.
- It shows:
  - A **percentage score** (0–100%) indicating how well you match the Tadasana template.
  - A green **“Success! Tadasana aligned”** banner when your score is high *and* the key conditions (feet together, arms by side, torso stacked, head straight) are all satisfied.

This means:

- If your legs are apart → no success.
- If your hands are away from your sides or lifted → no success.
- If your head is tilted noticeably → no success.

You can watch the score change as you adjust your posture and use it as a simple biofeedback tool to find a clean, balanced Tadasana.

---

## Requirements

- **Python**: 3.12 (or any version supported by your installed `mediapipe` wheel).
- **Hardware**: A webcam (built‑in or USB).
- **Python packages** (installed via `requirements.txt`):
  - `mediapipe`
  - `opencv-python`
  - `numpy`

---

## Setup

From the project root:

```bash
cd /Users/navnitnaman/Sthira

# 1. Create and activate a virtual environment (recommended)
python3.12 -m venv venv   # or python -m venv venv if 3.12 is your default
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> 💡 If you get errors about `mediapipe` not supporting your Python version, make sure you are using a supported Python (e.g. 3.10–3.12) inside the venv.

---

## Running the demo

With the virtual environment active:

```bash
cd /Users/navnitnaman/Sthira
python main.py
```

- A window titled **“MediaPipe Pose Demo”** will open and show your webcam feed.
- Colored skeleton lines appear over your body.
- At the top left you’ll see:
  - `Score: …` (overall Tadasana score)
  - A **green “Success! Tadasana aligned”** message when the pose matches the criteria.
- Press **`q`** to close the window and stop the program.

---

## Interpreting the feedback

While standing in front of the camera:

- **Feet together**: if your ankles are more than a few centimeters apart, your score drops and you won’t get success.
- **Arms by side**: if you raise your hands, move them forward/out, or let them hang far from your thighs, `arms_by_side` becomes false and success disappears.
- **Head straight**: if you tilt or turn your head, the `head_aligned` cue fails and your score decreases.

Use the score + success banner as a guide:

1. Stand straight with feet together.
2. Keep your weight equal on both feet.
3. Lengthen your spine and stack shoulders over hips.
4. Let your arms fall straight down by the sides of your body.
5. Keep your head straight, chin roughly parallel to the floor.

When you’re within the tolerance ranges for all these, you’ll see the success message.

---

## Project structure

```
Sthira/
├── main.py          # MediaPipe + OpenCV Tadasana detector & scorer
├── requirements.txt # Minimal dependency list (mediapipe, numpy, opencv-python)
└── README.md        # This documentation
```

---

## Where to go next

Now that the base Tadasana check is working, you can:

- **Adjust strictness** by tweaking the thresholds in `evaluate_tadasana` (e.g. foot distance, arm distance, head angle).
- **Add more cues** (e.g. pelvis neutrality, gaze direction, slight knee softness vs locked knees).
- **Add other simple poses** by writing new evaluation functions and toggles in `main.py`.
- **Layer audio feedback** back on top of this stable detector when you’re happy with the mechanics.

This focused setup should make it easier to iterate quickly and get to a truly robust, real‑time yoga trainer.  
When you’re ready, tell me which pose or feature you want to add next and we’ll build it step by step. 🎯🧘‍♂️