# 🧘 Sthira Project - Complete Explanation from Scratch

**A step-by-step guide to understanding the entire project**

---

## 📖 Table of Contents

1. [What is This Project?](#what-is-this-project)
2. [The Problem We're Solving](#the-problem-were-solving)
3. [How It Works - The Big Picture](#how-it-works---the-big-picture)
4. [Step-by-Step: What Happens When You Run It](#step-by-step-what-happens-when-you-run-it)
5. [The Training Process](#the-training-process)
6. [The Evaluation Process](#the-evaluation-process)
7. [File-by-File Breakdown](#file-by-file-breakdown)
8. [Data Flow Diagram](#data-flow-diagram)
9. [Key Concepts Explained](#key-concepts-explained)

---

## 🎯 What is This Project?

**Sthira** is a **real-time yoga pose evaluator** that:
- Uses your webcam to see your body
- Detects which yoga pose you're doing (out of 5 poses)
- Scores how well you're doing the pose (0-100%)
- Adapts to different body types automatically

**Think of it like:** A personal yoga instructor that watches you and gives instant feedback!

---

## 🤔 The Problem We're Solving

### The Challenge

**Traditional approach (mathematical rules):**
- Uses fixed thresholds (e.g., "feet must be 5cm apart")
- Doesn't work for different body types
- Tall person vs short person = different results for same pose
- Requires manual tuning for each pose

**Our solution (ML-based):**
- Learns what "ideal" looks like from real examples
- Adapts to different body sizes automatically
- Works for all body types
- One model handles multiple poses

### Why Body Adaptation Matters

Imagine:
- **Person A (5'2"):** Feet 4cm apart → passes threshold ✅
- **Person B (6'2"):** Feet 6cm apart → fails threshold ❌

But both are doing the pose correctly! Our ML model uses **relative measurements** instead of absolute distances.

---

## 🏗️ How It Works - The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    THE COMPLETE SYSTEM                      │
└─────────────────────────────────────────────────────────────┘

1. TRAINING PHASE (One-time, ~30 minutes)
   ┌──────────────────────────────────────┐
   │  Dataset (images of yoga poses)      │
   │  ↓                                   │
   │  MediaPipe extracts landmarks        │
   │  ↓                                   │
   │  Normalize by body size              │
   │  ↓                                   │
   │  Learn "ideal" pattern per pose      │
   │  ↓                                   │
   │  Save model (body_adaptive_pose_    │
   │              model.pkl)              │
   └──────────────────────────────────────┘

2. RUNTIME PHASE (Real-time, every frame)
   ┌──────────────────────────────────────┐
   │  Webcam captures frame               │
   │  ↓                                   │
   │  MediaPipe detects body landmarks    │
   │  ↓                                   │
   │  Extract features (normalized)       │
   │  ↓                                   │
   │  Compare to all 5 learned poses     │
   │  ↓                                   │
   │  Detect which pose (best match)      │
   │  ↓                                   │
   │  Score pose quality (0-100%)         │
   │  ↓                                   │
   │  Display on screen                   │
   └──────────────────────────────────────┘
```

---

## 🔄 Step-by-Step: What Happens When You Run It

### Phase 1: Startup (main.py)

**Step 1:** Program starts
```python
python main.py
```

**Step 2:** Load ML model
- Checks if `body_adaptive_pose_model.pkl` exists
- If yes: Loads it into memory
- If no: Falls back to math-based evaluation
- Loads list of poses from `pose_classes.txt`

**Step 3:** Initialize camera
- Opens webcam (default: camera 0)
- Sets up MediaPipe Pose detector

**Step 4:** Ready to run!

---

### Phase 2: Real-Time Loop (30+ times per second)

**For each frame from webcam:**

#### Step 1: Capture Frame
```
Webcam → Frame (image)
```

#### Step 2: Detect Body Landmarks
```
Frame → MediaPipe → 33 body landmarks
```

**What are landmarks?**
- 33 points on your body (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.)
- Each has: x, y, z coordinates + visibility score
- Example: `LEFT_SHOULDER = (0.45, 0.32, -0.1, 0.98)`

#### Step 3: Extract Features (Body-Adaptive)
```
33 landmarks → Extract relative features
```

**What happens here:**
1. Calculate body size (torso length)
2. Normalize all landmarks by body size
3. Extract 8 relative features:
   - Feet distance / shoulder width (ratio)
   - Shoulder offset / torso length
   - Arm positions / torso length
   - etc.
4. Add all 33 normalized landmarks
5. **Total: 141 features** (8 relative + 132 landmarks)

**Why normalize?**
- Person A: Torso = 0.15 units
- Person B: Torso = 0.10 units
- Same pose → Same normalized features!

#### Step 4: Detect Which Pose
```
Features → Compare to all 5 poses → Best match
```

**Process:**
1. Score current features against each pose:
   - Child's Pose: score = 0.45
   - Tree Pose: score = 0.82 ← **Best!**
   - Warrior I: score = 0.31
   - Downward Dog: score = 0.28
   - Plank: score = 0.15

2. Check if best score is good enough:
   - Best: 0.82
   - Second: 0.45
   - Gap: 0.37 (clear winner!)
   - **Result:** Tree Pose detected ✅

#### Step 5: Evaluate Pose Quality
```
Detected pose + Features → Score (0-100%)
```

**Scoring algorithm:**
1. Get ideal pattern for detected pose
2. Compare current features to ideal:
   - Euclidean distance in normalized space
   - Cosine similarity
   - Feature-by-feature comparison
3. Combine metrics with weights
4. Return score: 0.72 = 72%

#### Step 6: Display Results
```
Score + Pose name → Draw on screen
```

**What you see:**
- Skeleton overlay on your body
- "Pose: Tree Pose"
- "Score: 72.0%"
- "Good - Keep adjusting" (if score 60-80%)
- "✓ Excellent!" (if score > 80%)

#### Step 7: Repeat
- Goes back to Step 1
- Processes next frame
- 30+ times per second!

---

## 🎓 The Training Process

### What is Training?

**Training = Learning what "ideal" looks like from examples**

### Step-by-Step Training (train_body_adaptive.py)

#### Step 1: Load Dataset
```
data_set/train/
├── Child_Pose_or_Balasana_/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (184 images)
├── Tree_Pose_or_Vrksasana_/
│   └── ... (144 images)
└── ... (3 more poses)
```

#### Step 2: Process Each Image

**For each image:**
1. Load image
2. Extract landmarks using MediaPipe
3. Extract features (normalized by body size)
4. Store features

**Result:** Array of feature vectors for each pose

#### Step 3: Learn Ideal Pattern

**For each pose:**
1. Take all feature vectors from training images
2. Calculate **mean** (average) → This is the "ideal"
3. Fit StandardScaler → For normalization

**Example:**
- Child's Pose: 184 images → 184 feature vectors
- Mean of all vectors → 1 "ideal" feature vector
- This ideal = what perfect Child's Pose looks like

#### Step 4: Save Model

**Saves:**
- `ideal_poses`: Dictionary of ideal patterns (one per pose)
- `body_scalers`: StandardScalers for normalization (one per pose)

**File:** `body_adaptive_pose_model.pkl` (23 KB)

---

## 🔍 The Evaluation Process

### How Scoring Works (pose_evaluator.py)

#### Step 1: Extract Current Features
```
Your current pose → 141 features (normalized)
```

#### Step 2: Transform to Normalized Space
```
Features → StandardScaler → Normalized features
```

**Why?**
- Different features have different scales
- Normalize so all features are comparable
- Mean = 0, Std = 1

#### Step 3: Compare to Ideal

**Three metrics:**

**1. Euclidean Distance:**
```
distance = ||current_features - ideal_features||
score = exp(-distance / 8.0)
```
- Closer = higher score
- Exponential decay for smooth scoring

**2. Cosine Similarity:**
```
similarity = dot(current, ideal) / (||current|| * ||ideal||)
score = (similarity + 1) / 2
```
- Measures angle between vectors
- 1.0 = same direction, 0.0 = opposite

**3. Feature Comparison:**
```
For key features (feet, arms, head, etc.):
  diff = |current[i] - ideal[i]|
  score[i] = exp(-diff / 2.0)
Overall = mean of key feature scores
```
- Emphasizes important body parts

#### Step 4: Combine Metrics
```
Final Score = (Distance × 0.45) + (Cosine × 0.25) + (Features × 0.30)
```

**Result:** Score between 0.0 and 1.0 (0-100%)

---

## 📁 File-by-File Breakdown

### 1. `main.py` - The Main Application

**What it does:**
- Entry point of the program
- Handles webcam, displays results
- Coordinates everything

**Key Functions:**
- `load_ml_model()` - Loads trained model at startup
- `detect_pose_class()` - Detects which pose you're doing
- `evaluate_pose_ml()` - Scores pose quality using ML
- `run()` - Main loop (captures frames, processes, displays)
- `main()` - Entry point

**Flow:**
```
main() → load_ml_model() → run() → [loop: detect + evaluate + display]
```

---

### 2. `pose_evaluator.py` - The ML Brain

**What it does:**
- Contains the ML evaluation logic
- Handles body-adaptive feature extraction
- Scores poses against ideal patterns

**Key Class:**
- `BodyAdaptivePoseEvaluator` - The ML model

**Key Methods:**
- `normalize_by_body_size()` - Normalizes landmarks by torso length
- `extract_relative_features()` - Extracts 141 body-adaptive features
- `score_pose()` - Scores current pose against ideal (0-1)
- `load()` - Loads trained model from file

**How it works:**
```
Landmarks → Normalize → Extract Features → Compare to Ideal → Score
```

---

### 3. `train_body_adaptive.py` - The Trainer

**What it does:**
- Trains the ML model from dataset
- Processes all training images
- Learns ideal patterns
- Saves model

**Key Process:**
```
For each pose:
  1. Load all images
  2. Extract features from each
  3. Calculate mean (ideal pattern)
  4. Fit StandardScaler
  5. Save to model file
```

**Output:**
- `body_adaptive_pose_model.pkl` - Trained model
- `pose_classes.txt` - List of poses

---

### 4. `test_model.py` - The Tester

**What it does:**
- Tests trained model on test dataset
- Calculates accuracy
- Generates confusion matrix
- Creates test report

**Process:**
```
For each test image:
  1. Extract landmarks
  2. Detect pose (using model)
  3. Compare to true label
  4. Record result
Calculate: accuracy, confusion matrix, scores
```

---

### 5. Supporting Files

**`requirements.txt`** - Python packages needed
- mediapipe - Pose detection
- opencv-python - Camera and image processing
- numpy - Math operations
- scikit-learn - ML tools (StandardScaler)
- scipy - Scientific computing
- joblib - Model saving/loading

**`body_adaptive_pose_model.pkl`** - Trained model (23 KB)
- Contains ideal patterns for 5 poses
- Contains scalers for normalization

**`pose_classes.txt`** - List of trained poses
- One pose name per line

**`FINAL_REPORT_CARD.md`** - Test results and documentation

---

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW                       │
└─────────────────────────────────────────────────────────────┘

TRAINING (One-time):
────────────────────
Dataset Images
    ↓
[MediaPipe] Extract 33 landmarks per image
    ↓
[normalize_by_body_size] Normalize by torso length
    ↓
[extract_relative_features] Extract 141 features
    ↓
[Calculate mean] → Ideal pattern per pose
    ↓
[Fit StandardScaler] → Normalization scaler per pose
    ↓
[Save] → body_adaptive_pose_model.pkl

─────────────────────────────────────────────────────────────

RUNTIME (Real-time, 30+ FPS):
──────────────────────────────
Webcam Frame
    ↓
[MediaPipe] Extract 33 landmarks
    ↓
[normalize_by_body_size] Normalize by torso length
    ↓
[extract_relative_features] Extract 141 features
    ↓
[For each of 5 poses:]
    [StandardScaler.transform] Normalize features
    [score_pose] Compare to ideal
    → Get score (0-1)
    ↓
[detect_pose_class] Find best match
    ↓
[evaluate_pose_ml] Score the detected pose
    ↓
[Display] Show pose name + score on screen
    ↓
[Repeat] Next frame
```

---

## 🧠 Key Concepts Explained

### 1. Body Landmarks (33 points)

**What:** MediaPipe detects 33 key points on your body

**Examples:**
- NOSE (point 0)
- LEFT_SHOULDER (point 11)
- RIGHT_WRIST (point 16)
- LEFT_ANKLE (point 27)
- etc.

**Each landmark has:**
- `x, y` - Position (0.0 to 1.0, normalized to image)
- `z` - Depth (relative)
- `visibility` - How confident (0.0 to 1.0)

---

### 2. Body Normalization

**Problem:** Different people have different body sizes

**Solution:** Use torso length as reference unit

**How:**
```
1. Calculate torso length = distance(shoulder_center, hip_center)
2. Normalize all landmarks: (x - hip_center) / torso_length
3. Now all measurements are relative to body size!
```

**Example:**
- Tall person: Torso = 0.15 → Feet distance = 0.06 → Ratio = 0.4
- Short person: Torso = 0.10 → Feet distance = 0.04 → Ratio = 0.4
- **Same ratio = Same relative pose!** ✅

---

### 3. Feature Extraction

**What:** Convert 33 landmarks into 141 meaningful numbers

**Types of features:**

**A. Relative Features (9):**
- Feet distance / shoulder width
- Arm offsets / torso length
- Head alignment / torso length
- etc.

**B. Normalized Landmarks (132):**
- All 33 landmarks normalized
- Each has: x, y, z, visibility = 4 values
- 33 × 4 = 132 values

**Total:** 9 + 132 = 141 features

---

### 4. Ideal Pattern Learning

**What:** The "average" of all good examples

**How:**
```
Training images: [img1, img2, img3, ..., img184]
Features:        [feat1, feat2, feat3, ..., feat184]

Ideal = mean([feat1, feat2, feat3, ..., feat184])
```

**Result:** One feature vector representing "perfect" pose

**Why it works:**
- Good examples cluster together
- Mean = center of the cluster
- Represents typical "ideal" pose

---

### 5. Scoring Algorithm

**Three metrics combined:**

**A. Euclidean Distance:**
- Measures how far current pose is from ideal
- Uses exponential decay: `exp(-distance/8.0)`
- Closer = higher score

**B. Cosine Similarity:**
- Measures angle between feature vectors
- Same direction = high score
- Different direction = low score

**C. Feature Comparison:**
- Compares key body parts individually
- Emphasizes important features (feet, arms, head)
- Average of key feature scores

**Final Score:**
```
score = (distance_score × 0.45) + (cosine_score × 0.25) + (feature_score × 0.30)
```

---

### 6. Pose Detection

**How we know which pose:**

1. Score current pose against ALL 5 poses
2. Find best match (highest score)
3. Check if it's good enough:
   - Score > 25% minimum
   - Clear winner (gap > 5%) OR very high score (>60%)
4. Return pose name if confident

**Example:**
```
Scores: [Child: 0.45, Tree: 0.82, Warrior: 0.31, Dog: 0.28, Plank: 0.15]
Best: Tree (0.82)
Second: Child (0.45)
Gap: 0.37 (clear winner!)
Result: "Tree_Pose_or_Vrksasana_" ✅
```

---

## 🎬 Complete Example: User Doing Tree Pose

### Frame 1: User starts moving

1. **Webcam captures:** Image of person
2. **MediaPipe detects:** 33 landmarks (body visible)
3. **Extract features:** 141 normalized features
4. **Compare to poses:**
   - Child: 0.35
   - Tree: 0.42
   - Warrior: 0.38
   - Dog: 0.31
   - Plank: 0.25
5. **Detect:** Tree (best, but gap small)
6. **Score:** 0.42 = 42%
7. **Display:** "Pose: Tree Pose | Score: 42% | Adjust your pose"

### Frame 2: User improves pose

1. **Same process...**
2. **Compare:**
   - Child: 0.28
   - Tree: 0.78 ← Much better!
   - Warrior: 0.35
   - Dog: 0.29
   - Plank: 0.22
3. **Detect:** Tree (clear winner, gap = 0.50)
4. **Score:** 0.78 = 78%
5. **Display:** "Pose: Tree Pose | Score: 78.0% | Good - Keep adjusting"

### Frame 3: User perfects pose

1. **Same process...**
2. **Compare:**
   - Child: 0.25
   - Tree: 0.92 ← Excellent!
   - Warrior: 0.30
   - Dog: 0.27
   - Plank: 0.20
3. **Detect:** Tree (very high score)
4. **Score:** 0.92 = 92%
5. **Display:** "Pose: Tree Pose | Score: 92.0% | ✓ Excellent!"

**All happening 30+ times per second!**

---

## 🔑 Why This Approach Works

### 1. Body Adaptation ✅
- Uses relative measurements, not absolute
- Works for tall, short, wide, narrow people
- No manual tuning needed

### 2. Learning from Data ✅
- Learns what "ideal" means from real examples
- Adapts to your dataset
- More examples = better model

### 3. Real-time Performance ✅
- Fast feature extraction
- Efficient comparison (just math operations)
- No heavy neural networks

### 4. Interpretable ✅
- Can see which features matter
- Understand why score is low
- Easy to debug

---

## 📊 Performance Summary

**Accuracy:** 93.33% on test dataset
- Tree Pose: 100%
- Plank: 100%
- Child's Pose: 95.83%
- Warrior I: 87.50%
- Downward Dog: 85.71%

**Speed:** 30+ FPS (real-time)

**Model Size:** 23 KB (very efficient!)

**Body Types:** Works for all (adaptive)

---

## 🚀 How to Use

### 1. Train Model (One-time)
```bash
python train_body_adaptive.py
```
- Processes ~636 training images
- Takes ~30 minutes
- Creates `body_adaptive_pose_model.pkl`

### 2. Run Application
```bash
python main.py
```
- Loads model automatically
- Opens webcam
- Shows real-time feedback

### 3. Test Model (Optional)
```bash
python test_model.py
```
- Tests on test dataset
- Shows accuracy metrics

---

## 💡 Key Takeaways

1. **MediaPipe** detects your body → 33 landmarks
2. **Feature extraction** converts landmarks → 141 features (body-adaptive)
3. **Pose detection** compares features → finds best match
4. **Scoring** compares to ideal → gives 0-100% score
5. **Display** shows results → real-time feedback

**The magic:** Body normalization makes it work for everyone!

---

## 🎓 Summary

**What:** Real-time yoga pose evaluator with ML

**How:** 
- Train once on dataset → Learn ideal patterns
- Run anytime → Compare live pose to ideals → Score

**Why ML instead of math:**
- Adapts to different body types
- Learns from data
- More accurate

**Result:** 93.33% accuracy, real-time, body-adaptive! ✅

---

*This explanation covers the entire project from scratch. Every concept, every step, every file is explained in sequence.*

