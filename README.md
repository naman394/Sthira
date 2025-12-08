# 🧘 Sthira - Body-Adaptive Yoga Pose Evaluator

A real-time yoga pose evaluation system that uses **machine learning** to detect and score yoga poses, automatically adapting to different body types. Built with [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose), OpenCV, and a custom ML model.

---

## ✨ Features

- **🎯 Multi-Pose Detection**: Automatically detects 5 yoga poses in real-time
- **📊 Quality Scoring**: Provides 0-100% score for pose quality
- **🔄 Body-Adaptive**: Automatically adapts to different body types and sizes
- **⚡ Real-Time Performance**: Runs at 30+ FPS on standard hardware
- **📈 High Accuracy**: 93.33% accuracy on test dataset
- **🎓 ML-Powered**: Uses template matching with statistical learning

### Supported Poses

1. **Child's Pose** (Balasana)
2. **Tree Pose** (Vrksasana)
3. **Warrior I** (Virabhadrasana I)
4. **Downward-Facing Dog** (Adho Mukha Svanasana)
5. **Plank Pose** (Kumbhakasana)

---

## 🎬 What It Does

When you run the application with a webcam:

1. **Real-Time Detection**: Uses MediaPipe to track 33 body landmarks
2. **Pose Classification**: Automatically identifies which of the 5 poses you're doing
3. **Quality Evaluation**: Scores your pose (0-100%) using body-adaptive ML model
4. **Visual Feedback**: 
   - Skeleton overlay on your body
   - Detected pose name
   - Quality score with color-coded feedback
   - Detailed alignment cues

### How Body Adaptation Works

The model normalizes all measurements by body size (torso length), so it works equally well for:
- Tall and short people
- Different body proportions
- Various camera distances
- Different body types

**Example**: A tall person and short person doing the same pose will get similar scores because the model uses relative measurements, not absolute distances.

---

## 📋 Requirements

- **Python**: 3.10-3.12 (recommended: 3.12)
- **Hardware**: Webcam (built-in or USB)
- **Python Packages** (see `requirements.txt`):
  - `mediapipe>=0.10.14` - Pose detection
  - `opencv-python>=4.8.0` - Camera and image processing
  - `numpy>=1.24.0` - Numerical operations
  - `scikit-learn>=1.3.0` - ML tools (StandardScaler)
  - `joblib>=1.3.0` - Model serialization
  - `scipy>=1.11.0` - Scientific computing

---

## 🚀 Installation & Quick Start

### Step 1: Extract the Project

1. Extract the `Sthira_Clean.zip` file to your desired location
2. Open a terminal/command prompt and navigate to the extracted folder:
   ```bash
   cd /path/to/Sthira
   ```

### Step 2: Check Python Version

Make sure you have Python 3.10, 3.11, or 3.12 installed:

```bash
python3 --version
# or
python --version
```

**Required**: Python 3.10-3.12 (MediaPipe requirement)

### Step 3: Create Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt after activation.

### Step 4: Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- `mediapipe` - Pose detection
- `opencv-python` - Camera and image processing
- `numpy` - Numerical operations
- `scikit-learn` - ML tools
- `joblib` - Model serialization
- `scipy` - Scientific computing

**Installation time**: ~2-5 minutes depending on your internet speed.

> ⚠️ **Troubleshooting**: If you get errors about `mediapipe` not supporting your Python version:
> - Ensure you're using Python 3.10-3.12
> - Make sure the virtual environment is activated
> - Try: `pip install --upgrade pip` first, then `pip install -r requirements.txt`

### Step 5: Verify Installation

Check that all packages are installed correctly:

```bash
python -c "import mediapipe, cv2, numpy, sklearn, joblib, scipy; print('✅ All packages installed successfully!')"
```

### Step 6: Run the Application

The project includes a pre-trained model (`body_adaptive_pose_model.pkl`), so you can run it immediately:

```bash
python main.py
```

**Or specify a different camera source:**
```bash
python main.py --source 1  # Use camera index 1 instead of 0
```

**What you'll see:**
- Webcam window with skeleton overlay
- Detected pose name (e.g., "Tree Pose")
- Quality score (0-100%)
- Color-coded feedback (red/yellow/green)
- Alignment cues

**Press `q` to quit the application.**

**What you'll see:**
- Webcam window with skeleton overlay
- Detected pose name (e.g., "Tree Pose")
- Quality score (0-100%)
- Color-coded feedback (red/yellow/green)
- Alignment cues

Press **`q`** to quit.

---

## 🎓 Training Your Own Model

If you want to train the model on your own dataset:

### 1. Prepare Your Dataset

Organize your images in this structure:
```
data_set/
└── train/
    ├── Child_Pose_or_Balasana_/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Tree_Pose_or_Vrksasana_/
    │   └── ...
    └── ... (other poses)
```

### 2. Train the Model

```bash
python train_body_adaptive.py
```

This will:
- Process all training images
- Extract body-adaptive features
- Learn ideal patterns for each pose
- Save model to `body_adaptive_pose_model.pkl`

**Training time**: ~30 minutes for ~636 images

### 3. Test the Model

```bash
python test_model.py
```

This evaluates the model on the test dataset and generates a detailed report.

**See `TRAINING_GUIDE.md` for detailed training instructions.**

---

## 📊 Model Performance

### Test Results

- **Overall Accuracy**: **93.33%** (98/105 correct predictions)
- **Per-Pose Accuracy**:
  - Tree Pose: **100%** ✅
  - Plank Pose: **100%** ✅
  - Child's Pose: **95.83%** ✅
  - Warrior I: **87.50%** ✅
  - Downward Dog: **85.71%** ✅

### Model Specifications

- **Model Size**: 23 KB (very efficient!)
- **Inference Speed**: 30+ FPS (real-time)
- **Training Data**: ~636 images
- **Test Data**: ~105 images
- **Features**: 141 body-adaptive features per pose

**See `FINAL_REPORT_CARD.md` for complete test results and analysis.**

---

## 📁 Project Structure

```
Sthira/
├── main.py                      # Main application (real-time pose evaluation)
├── pose_evaluator.py            # ML model (BodyAdaptivePoseEvaluator)
├── train_body_adaptive.py        # Training script
├── test_model.py                # Testing script
├── body_adaptive_pose_model.pkl # Trained model (23 KB)
├── pose_classes.txt             # List of supported poses
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── TRAINING_GUIDE.md            # Training instructions
├── PROJECT_EXPLANATION.md       # Complete project explanation
├── FINAL_REPORT_CARD.md         # Test results and analysis
└── data_set/                    # Dataset (not in repo, too large)
    ├── train/                   # Training images
    ├── test/                    # Test images
    └── valid/                   # Validation images
```

---

## 🔧 How It Works

### Architecture

```
Webcam Frame
    ↓
MediaPipe Pose (Pre-trained)
    ↓
33 Body Landmarks
    ↓
BodyAdaptivePoseEvaluator (Custom ML)
    ├── Extract 141 features (body-normalized)
    ├── Compare to 5 learned ideal patterns
    ├── Detect pose (best match)
    └── Score quality (0-100%)
    ↓
Display: Pose Name + Score + Feedback
```

### Model Type

The custom model uses:
- **Template Matching**: Compares current pose to learned ideal templates
- **Statistical Learning**: Ideal pattern = mean of training features
- **Similarity Metrics**: Euclidean distance, cosine similarity, feature comparison
- **Body Normalization**: All features normalized by body size (torso length)

**Not a neural network** - it's a lightweight, interpretable ML approach that's fast and efficient.

---

## 📖 Documentation

- **`PROJECT_EXPLANATION.md`**: Complete explanation of the entire project from scratch
- **`TRAINING_GUIDE.md`**: Step-by-step training instructions
- **`FINAL_REPORT_CARD.md`**: Detailed test results and performance analysis

---

## 🎯 Usage Examples

### Basic Usage
```bash
python main.py
```

### Use Different Camera
```bash
python main.py --source 1
```

### Train Model
```bash
python train_body_adaptive.py
```

### Test Model
```bash
python test_model.py
```

---

## 🛠️ Troubleshooting

### Model Not Found
If you see: `⚠️ ML model not found`
- The model file (`body_adaptive_pose_model.pkl`) should be in the project root
- If missing, run: `python train_body_adaptive.py`

### Camera Not Working
- Check camera permissions
- Try different camera index: `python main.py --source 1`
- Ensure no other app is using the camera

### Poor Detection
- Ensure good lighting
- Stand 2-3 meters from camera
- Make sure full body is visible
- Avoid cluttered backgrounds

### Import Errors
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt`
- Check Python version (3.10-3.12)

---

## 🚀 Deployment

### Local Desktop Application

Runs on any computer with:
- Python 3.10-3.12
- Webcam
- Dependencies from `requirements.txt`

**Package as executable** (optional):
```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

### Web Deployment (Future)

For web deployment, you would need to:
- Convert OpenCV to use WebRTC for browser camera access
- Run MediaPipe in the browser using MediaPipe JavaScript
- Or set up a server with video streaming

### Cloud Deployment

For cloud deployment (e.g., AWS, Google Cloud):
- Requires video streaming infrastructure
- Consider using MediaPipe in the cloud with WebRTC
- Or use MediaPipe's web-based solutions

---

## 🔮 Future Enhancements

Potential improvements:
- ✅ More poses (currently 5, easily expandable)
- ✅ Pose sequences/workouts
- ✅ Session recording and replay
- ✅ Progress tracking over time
- ✅ Audio feedback
- ✅ Better UI with progress bars and visualizations
- ✅ Mobile app support

---

## 📝 License

This project is open source. See repository for license details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more poses
- Improve the UI
- Optimize performance
- Add new features

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for pose detection
- **OpenCV** for computer vision
- **scikit-learn** for ML tools

---

**Built with ❤️ for yoga practitioners**

*Sthira* - Sanskrit for "steady" or "stable" - reflecting the goal of achieving steady, well-aligned yoga poses.
