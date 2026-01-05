# 🏏 AI Third Umpire - LBW Detection System

An advanced AI-powered third umpire system for cricket that uses computer vision, machine learning, and predictive analytics to detect LBW (Leg Before Wicket) decisions accurately. This system reduces human error and eliminates the need for 'umpire's call' by providing precise, data-driven decisions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Training LSTM Model](#-training-lstm-model)
- [Evaluation Metrics](#-evaluation-metrics)
- [Understanding Model Files](#-understanding-model-files)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Technologies Used](#-technologies-used)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## ✨ Features

- **🎯 Accurate Ball Detection**: Uses YOLOv8 for real-time ball detection and tracking
- **📊 Advanced Trajectory Analysis**: LSTM-based trajectory prediction with polynomial fallback
- **🧠 ML-Based Classification**: XGBoost classifier for LBW decision making (with automatic rule-based fallback)
- **🎬 3D Visualization**: Interactive 3D trajectory plots using Plotly
- **📄 PDF Reports**: Automated generation of detailed technical reports
- **📈 Analytics Dashboard**: Comprehensive analytics and review history
- **🤖 AI-Powered Explanations**: Gemini AI integration for detailed decision explanations
- **🎥 Video Replay**: Generate annotated replay videos with trajectory overlays
- **⚡ Real-time Analysis**: Fast processing with configurable parameters

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/iamDeepakHR/Third_Umpire_LBW_Detection_Using_AI.git
cd Third_Umpire_LBW_Detection_Using_AI
```

### Step 2: Install Dependencies

**Option A: Using setup script (Recommended)**

```bash
python setup.py
```

**Option B: Manual installation**

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

The YOLOv8 weights file (`yolov8n.pt`) should already be included. If not, it will be automatically downloaded on first run.

### Step 4: (Optional) Train LSTM Model

The system works without trained models (uses fallbacks), but for better accuracy:

```bash
python train_lstm_trajectory.py --synthetic --num-synthetic 1000 --epochs 50
```

See [Training LSTM Model](#-training-lstm-model) section for details.

### Step 5: (Optional) Get Gemini API Key

For AI-powered explanations, get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and configure it in the application sidebar.

## ⚡ Quick Start

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload a video or select from samples:**
   - Click "Upload a cricket video" or choose from demo videos
   - Click "🚀 Run LBW Analysis"

3. **View results:**
   - See the decision (OUT/NOT OUT) with confidence
   - Explore 3D visualizations and detailed analysis
   - Download PDF reports

**That's it!** The system works out of the box with automatic fallback methods. No model files needed!

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Application Features

1. **Video Analysis Tab:**
   - Upload cricket videos (MP4, MOV, AVI, MKV)
   - Select from sample videos
   - View real-time analysis with trajectory overlays
   - Generate PDF reports and replay videos

2. **Analytics Dashboard:**
   - View statistics on all reviews
   - See decision distributions
   - Track confidence scores over time

3. **History Tab:**
   - Review past analyses
   - Access detailed information for each review

### Command Line Usage

```python
from lbw_ai.detector import BallDetector
from lbw_ai.tracker import SimpleBallTracker
from lbw_ai.trajectory import fit_trajectory
from lbw_ai.classifier import LBWClassifier

# Initialize components
detector = BallDetector(weights_path="yolov8n.pt")
tracker = SimpleBallTracker()

# Classifier works automatically - no model file needed!
# Uses rule-based fallback if models/lbw_xgb.json doesn't exist
classifier = LBWClassifier("models/lbw_xgb.json")

# Process video and get trajectory
# Uses LSTM if models/lstm_trajectory.pth exists, else polynomial
trajectory = fit_trajectory(track_points, model_path="models/lstm_trajectory.pth")
future_points = trajectory.predict(num_future=15)
```

## 🎓 Training LSTM Model

The LSTM model improves trajectory prediction accuracy. You can train it with synthetic data (no videos needed) or extract trajectories from your videos.

### Quick Start: Train with Synthetic Data (No Videos Needed!)

**Easiest method - no data required:**

```bash
python train_lstm_trajectory.py --synthetic --num-synthetic 1000 --epochs 50
```

**What this does:**
- Generates 1000 synthetic ball trajectories automatically
- Trains LSTM model for 50 epochs (~5-10 minutes on CPU)
- Saves model to `models/lstm_trajectory.pth`
- The app will automatically use it!

### Training Options

**Quick Test (2-3 minutes):**
```bash
python train_lstm_trajectory.py --synthetic --num-synthetic 500 --epochs 20
```

**Better Accuracy (10-15 minutes):**
```bash
python train_lstm_trajectory.py --synthetic --num-synthetic 2000 --epochs 100
```

**Maximum Accuracy (30-60 minutes):**
```bash
python train_lstm_trajectory.py --synthetic --num-synthetic 5000 --epochs 200 --hidden-size 128
```

### Training with Real Video Data

If you want to train on real trajectories from your videos:

**Step 1: Extract trajectories from videos**
```bash
python extract_trajectories_from_videos.py samples/ --output trajectories.json
```

**Step 2: Train on extracted data**
```bash
python train_lstm_trajectory.py --data-file trajectories.json --epochs 100
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--synthetic` | Generate synthetic training data | - |
| `--num-synthetic` | Number of synthetic trajectories | 1000 |
| `--data-file` | Path to trajectory JSON file | - |
| `--epochs` | Number of training epochs | 50 |
| `--batch-size` | Batch size for training | 32 |
| `--learning-rate` | Learning rate | 0.001 |
| `--hidden-size` | LSTM hidden layer size | 64 |
| `--num-layers` | Number of LSTM layers | 2 |
| `--sequence-length` | Input sequence length | 10 |
| `--output` | Output model path | models/lstm_trajectory.pth |
| `--device` | Device (cpu/cuda/auto) | auto |

### Understanding Training Data

**Important:** The LSTM trains on **trajectory coordinates** (x, y points), NOT directly on videos or images.

- **Synthetic data**: Automatically generated realistic trajectories
- **Real data**: Extract from videos using `extract_trajectories_from_videos.py`
- **Format**: JSON file with list of trajectories, each trajectory is list of [x, y] coordinates

**Example trajectory data format:**
```json
[
  [[100, 200], [105, 210], [110, 220], [115, 230], ...],
  [[150, 180], [155, 190], [160, 200], [165, 210], ...]
]
```

## 📊 Evaluation Metrics

For IEEE-ready evaluation, use `evaluate_classifier.py` to compute all standard classification metrics and generate high-resolution figures.

### Step 1: Create Evaluation Dataset

Prepare a CSV file containing the ground-truth LBW decisions and the model's predictions (see `sample_lbw_results.csv` for a ready-to-use example):

```csv
actual,predicted
OUT,OUT
NOT OUT,NOT OUT
OUT,NOT OUT
...
```

You can export these labels from match annotations, integration tests, or logged review history. Values must be `OUT` or `NOT OUT`.

### Step 2: Run the Evaluation Script

```bash
python evaluate_classifier.py --data-file my_lbw_results.csv --output-dir outputs/evaluation
```

The script prints the metrics to the terminal and generates:

- `outputs/evaluation/metrics_summary.json` – ready to cite in your paper or supplemental material
- `outputs/evaluation/confusion_matrix.png` – high-resolution 300 dpi figure for IEEE manuscripts or presentations

### Metrics Reported

| Metric | Description |
|--------|-------------|
| **True Positive (TP)** | Actual OUT correctly predicted as OUT |
| **True Negative (TN)** | Actual NOT OUT correctly predicted as NOT OUT |
| **False Positive (FP)** | Actual NOT OUT incorrectly predicted as OUT |
| **False Negative (FN)** | Actual OUT incorrectly predicted as NOT OUT |
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` |
| **Precision** | `TP / (TP + FP)` |
| **Recall / Sensitivity** | `TP / (TP + FN)` |
| **Specificity** | `TN / (TN + FP)` |
| **F1-Score** | `2 * (Precision * Recall) / (Precision + Recall)` |
| **False Positive Rate (FPR)** | `FP / (FP + TN)` |
| **False Negative Rate (FNR)** | `FN / (FN + TP)` |

Include the exported JSON table and the generated confusion-matrix image in your IEEE research paper to document classifier performance comprehensively.

## 📊 Understanding Model Files

### ⚠️ Important: Model Files Are Optional!

The system **works perfectly without any model files** using automatic fallbacks. Model files only improve accuracy but are **not required**.

### 1. LSTM Trajectory Model (`models/lstm_trajectory.pth`)

**Status:** Optional - Train using `train_lstm_trajectory.py`

**What it does:** Improves trajectory prediction accuracy for complex ball paths

**Fallback:** Polynomial curve fitting (always available)

**How to get it:**
```bash
python train_lstm_trajectory.py --synthetic --num-synthetic 1000 --epochs 50
```

**Usage:** System automatically uses LSTM if file exists, falls back to polynomial if not.

### 2. XGBoost Classifier (`models/lbw_xgb.json`)

**Status:** Optional - Not included in repository

**What it does:** Improves LBW decision classification accuracy

**Fallback:** Rule-based classifier (automatic - no setup needed!)

**How it works:**
```python
# This code works even if models/lbw_xgb.json doesn't exist!
classifier = LBWClassifier("models/lbw_xgb.json")

# System automatically checks:
# - If file exists → Uses XGBoost model
# - If file doesn't exist → Uses rule-based fallback automatically
```

**Rule-based fallback scoring:**
- Pitched in-line: +0.4 points
- Impact in-line: +0.6 points
- Would hit stumps: +0.8 points
- Distance penalty: -0.3 points (based on distance)
- Decision: OUT if score ≥ 0.5, otherwise NOT OUT

**You don't need to do anything!** The system works automatically.

### Summary

| Model File | Required? | Fallback | How to Get |
|------------|-----------|----------|------------|
| `lstm_trajectory.pth` | ❌ No | Polynomial fitting | Train with `train_lstm_trajectory.py` |
| `lbw_xgb.json` | ❌ No | Rule-based heuristics | Train on your dataset (optional) |

**Bottom line:** The system works out of the box! Model files are optional enhancements.

## 📁 Project Structure

```
Third_Umpire_LBW_Detection_Using_AI/
│
├── app.py                          # Main Streamlit application
├── setup.py                        # Setup script for installation
├── requirements.txt                # Python dependencies
├── test_integration.py             # Integration tests
├── train_lstm_trajectory.py        # LSTM model training script
├── evaluate_classifier.py          # Classification metrics & plots
├── extract_trajectories_from_videos.py  # Extract trajectories from videos
├── yolov8n.pt                      # YOLOv8 model weights
├── README.md                       # This file
│
├── lbw_ai/                         # Core AI module
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   ├── detector.py                # Ball detection using YOLO
│   ├── tracker.py                 # Ball tracking algorithm
│   ├── bounce.py                  # Bounce point detection
│   ├── trajectory.py              # Trajectory prediction (LSTM + Polynomial)
│   ├── impact.py                  # Impact and stumps prediction
│   ├── classifier.py              # LBW decision classifier (XGBoost + Rule-based)
│   ├── visualize.py               # 2D visualization
│   ├── visualize_3d.py            # 3D trajectory plots
│   ├── explainer.py               # AI-powered explanations
│   ├── analytics.py               # Analytics dashboard
│   └── report_generator.py        # PDF report generation
│
├── samples/                        # Sample cricket videos
│   ├── lbw.mp4
│   ├── lbw1.mp4
│   └── ...
│
├── models/                         # ML model files (optional)
│   ├── README.md                  # Model documentation
│   ├── lbw_xgb.json               # XGBoost classifier (optional)
│   └── lstm_trajectory.pth        # LSTM trajectory model (optional)
│
├── outputs/                        # Generated outputs
│   ├── *_report.pdf
│   └── *_overlay.mp4
│
└── analytics/                      # Analytics data
    └── reviews.json
```

## 🔧 How It Works

### 1. Ball Detection
- Uses YOLOv8 object detection model to identify the cricket ball in each frame
- Filters detections based on confidence and IoU thresholds

### 2. Ball Tracking
- Tracks the ball across frames using a simple tracking algorithm
- Maintains trajectory history for analysis

### 3. Bounce Detection
- Analyzes trajectory changes to detect bounce points
- Identifies where the ball hits the pitch

### 4. Trajectory Prediction
- **Primary**: LSTM-based sequence-to-sequence prediction (if model available)
  - Learns complex trajectory patterns from training data
  - Better handles non-linear movements and bounce effects
  - Improved accuracy for complex trajectories
- **Fallback**: Polynomial curve fitting
  - Fits polynomial curves to the ball's trajectory
  - Predicts future ball path using mathematical modeling
  - Always available as backup method

### 5. Impact Analysis
- Estimates stumps region in the frame
- Predicts whether the ball would hit the stumps
- Calculates distance to stumps if it misses

### 6. LBW Classification
- Extracts features: pitched zone, impact in-line, would hit stumps, distance
- **Primary**: Uses XGBoost classifier (if model file exists)
- **Fallback**: Rule-based heuristics (automatic if model file missing)
  - Scoring system based on key LBW factors
  - Provides confidence score for the decision
- Returns decision: OUT or NOT OUT with confidence percentage

### 7. Visualization & Reporting
- Generates 2D and 3D trajectory visualizations
- Creates annotated video replays
- Generates comprehensive PDF reports
- Provides AI-powered explanations

## 🛠️ Technologies Used

- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Machine Learning**: XGBoost, scikit-learn, PyTorch
- **Deep Learning**: LSTM for trajectory prediction
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib, Pillow
- **AI Explanations**: Google Generative AI (Gemini)
- **Data Processing**: NumPy, Pandas, SciPy
- **Report Generation**: ReportLab (via report_generator)

## ⚙️ Configuration

Edit `lbw_ai/config.py` or use the Streamlit sidebar to configure:

- **YOLO Weights Path**: Path to YOLOv8 model weights
- **Confidence Threshold**: Minimum confidence for ball detection (default: 0.25)
- **IoU Threshold**: Intersection over Union threshold (default: 0.45)
- **Max Frames**: Limit number of frames to process (None for all)
- **LSTM Model Path**: Path to LSTM trajectory model (default: models/lstm_trajectory.pth)
- **Use LSTM**: Enable/disable LSTM trajectory prediction (default: True)
- **XGBoost Model**: Path to trained classifier model (default: models/lbw_xgb.json)
- **Stumps Position**: Configure stumps region in frame

## 🧪 Testing

Run the integration tests:

```bash
python test_integration.py
```

This will verify:
- All required modules can be imported
- Basic explanation generation works
- Trajectory visualization functions correctly

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Write tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Deepak HR**

- GitHub: [@iamDeepakHR](https://github.com/iamDeepakHR)
- Repository: [Third_Umpire_LBW_Detection_Using_AI](https://github.com/iamDeepakHR/Third_Umpire_LBW_Detection_Using_AI)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- Streamlit for the web framework
- Google Generative AI for explanation features
- The cricket community for inspiration

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/iamDeepakHR/Third_Umpire_LBW_Detection_Using_AI/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

## 🔮 Future Enhancements

- [ ] Multi-camera support for better accuracy
- [ ] Real-time video stream processing
- [ ] Integration with broadcast systems
- [ ] Mobile app version
- [ ] Advanced physics modeling
- [ ] Player pose estimation
- [ ] Automated pitch calibration

---

**Note**: This is an AI-based system designed to assist in LBW decisions. For official cricket matches, always follow ICC regulations and use certified systems.

Made with ❤️ for cricket enthusiasts
