# Body-Adaptive Pose Training Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_body_adaptive.py
```

This will:
- Process images from `data_set/train/` for 5 poses
- Extract body-adaptive features
- Learn ideal pose patterns
- Save model to `body_adaptive_pose_model.pkl`

**Time estimate:** 15-45 minutes (depends on dataset size)

### 3. Run the App

```bash
python main.py
```

The app will automatically:
- Load the ML model if available
- Detect which pose you're doing
- Evaluate pose quality (body-adaptive)
- Fall back to math-based evaluation if model not found

## What Gets Trained

The model learns ideal patterns for these 5 poses:
1. Tadasana (Mountain Pose)
2. Tree Pose (Vrksasana)
3. Warrior I (Virabhadrasana I)
4. Downward-Facing Dog (Adho Mukha Svanasana)
5. Child's Pose (Balasana)

## How It Works

### Body Adaptation
- Normalizes by body size (torso length as reference)
- Uses relative measurements instead of absolute distances
- Adapts to different body types automatically

### Pose Detection
- Scores current pose against all learned poses
- Returns the best match (if confidence > 50%)

### Evaluation
- Compares current pose to ideal pattern
- Returns score 0-100% (how close to ideal)
- Adapts to your body proportions

## Files Created

- `body_adaptive_pose_model.pkl` - Trained model
- `pose_classes.txt` - List of trained poses
- `pose_evaluator.py` - ML evaluation module

## Troubleshooting

**Model not found?**
- Run `python train_body_adaptive.py` first
- Check that `data_set/train/` contains pose folders

**Pose not detected?**
- Make sure you're doing one of the 5 trained poses
- Ensure good lighting and full body visibility
- Try adjusting your pose to match the training data

**Low scores?**
- The model learns from your dataset - if dataset has good examples, scores will be accurate
- Body adaptation means it works for different body types

## Next Steps

- Add more poses: Edit `POSE_CLASSES` in `train_body_adaptive.py`
- Adjust thresholds: Modify confidence thresholds in `main.py`
- Improve accuracy: Add more training images to dataset

