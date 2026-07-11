# FER2013 Facial Emotion Classification

## Overview
FER2013 Facial Emotion Classification is an end-to-end computer vision project that trains, evaluates, exports, and serves a facial emotion classifier. The system predicts one of seven emotion classes from grayscale face images: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

The project solves a practical image-classification workflow problem: moving from raw Kaggle data to a trained TensorFlow model, evaluation artifacts, deployment-ready exports, and an interactive demo. It is suitable for machine learning portfolio review, educational experimentation, and lightweight prototype applications that need emotion-aware user interaction.

## Technology Stack
- **Language:** Python
- **Frameworks:** TensorFlow/Keras, Streamlit
- **Key Libraries:** TensorFlowJS, NumPy, Pandas, scikit-learn, Matplotlib, Pillow, tqdm, kagglehub
- **Dataset:** FER2013 from Kaggle (`msambare/fer2013`)
- **Model Formats:** Keras `.keras`, TensorFlow SavedModel, TensorFlow Lite, TensorFlow.js
- **Infrastructure:** Local training/inference pipeline; no cloud deployment or CI/CD is included in this repository

## Features
- Downloads and prepares FER2013 data through `kagglehub`.
- Creates reproducible `train`, `val`, and `test` directory splits.
- Trains a TensorFlow/Keras emotion classifier with transfer learning.
- Applies on-the-fly image augmentation for training data.
- Evaluates the best checkpoint on train, validation, and test sets.
- Generates accuracy/loss curves and confusion matrix visualizations.
- Exports trained models to SavedModel, TFLite, and TFJS formats.
- Provides command-line inference for SavedModel and TFLite models.
- Includes a Streamlit app for image upload or camera-based emotion prediction.

## Architecture
The project follows a modular ML pipeline: dataset preparation, model training, evaluation, export, and inference are separated into scripts under `submission/src`.

```text
Kaggle FER2013 Dataset
        |
        v
Data Preparation / Resplitting
        |
        v
TensorFlow Dataset Pipeline
(normalization + augmentation)
        |
        v
Keras Model Training
(CNN front-end + MobileNetV2 backbone + classifier head)
        |
        v
Evaluation Artifacts
(accuracy/loss plots + confusion matrix)
        |
        v
Model Export
(SavedModel + TFLite + TensorFlow.js)
        |
        v
Inference
(CLI scripts + Streamlit UI)
```

### Model Design
The classifier accepts `48x48x1` grayscale images. A lightweight convolutional front-end extracts local grayscale features, projects them to three channels, resizes them to `96x96x3`, and passes them through an ImageNet-pretrained MobileNetV2 backbone. A global average pooling layer and dense classification head produce seven-class softmax probabilities.

## Project Structure

```text
submission/
├── README.md
├── requirements.txt
├── notebook.ipynb
├── history.npy
├── data/
│   ├── raw/                 # Raw FER2013 mirror from Kaggle
│   ├── images/              # Original prepared image directory, if available
│   └── images_resplit/      # Train/validation/test image splits
├── plots/                   # Training curves, samples, confusion matrices
├── sample_images/           # Example images for inference
├── saved_model/             # Keras checkpoint and TensorFlow SavedModel
├── tflite/                  # TFLite model and labels
├── tfjs_model/              # TensorFlow.js model shards
└── src/
    ├── app.py
    ├── build_model.py
    ├── download_and_prepare.py
    ├── evaluate_and_plots.py
    ├── export_models.py
    ├── infer_savedmodel.py
    ├── infer_tflite.py
    ├── resplit_dataset.py
    ├── train.py
    └── utils.py
```

## Prerequisites
- Python 3.10+ recommended
- Kaggle access configured if downloading the dataset from Kaggle
- Enough local disk space for FER2013 images and exported model artifacts
- Optional: Node.js and `http-server` if serving the TensorFlow.js model locally

## Installation & Setup

### 1. Clone/Prepare

```bash
git clone <your-repository-url>
cd "Proyek Klasifikasi Gambar"
```

If you already have this project locally, run commands from the repository root.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r submission/requirements.txt
```

### 4. Prepare Dataset

```bash
python3 submission/src/download_and_prepare.py
```

This downloads `msambare/fer2013` via `kagglehub`, mirrors raw files under `submission/data/raw/msambare_fer2013`, and prepares train/validation/test folders under `submission/data/images_resplit`.

Optional local resplit:

```bash
python3 submission/src/resplit_dataset.py --force
```

### 5. Train

```bash
python3 submission/src/train.py
```

Useful options:

```bash
python3 submission/src/train.py \
  --epochs 15 \
  --fine-tune-epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --fine-tune-lr 1e-4
```

### 6. Evaluate

```bash
python3 submission/src/evaluate_and_plots.py
```

Evaluation writes plots to `submission/plots` and prints classification metrics to the terminal.

### 7. Export Models

```bash
python3 submission/src/export_models.py
```

This refreshes the SavedModel export and writes deployment artifacts to `submission/tflite` and `submission/tfjs_model`.

### 8. Run Streamlit Demo

```bash
streamlit run submission/src/app.py
```

## Usage Examples

### Example 1: TFLite Inference

```bash
python3 submission/src/infer_tflite.py --image submission/sample_images/happy.png
```

Output format:

```text
Prediction: happy
Confidence: 0.95
```

Exact confidence depends on the trained model checkpoint and input image.

### Example 2: SavedModel Inference

```bash
python3 submission/src/infer_savedmodel.py --image submission/sample_images/happy.png
```

Output format:

```text
Prediction: happy
Confidence: 0.95
```

### Example 3: Browser Demo

```bash
streamlit run submission/src/app.py
```

The app lets users upload an image or capture a camera snapshot, then displays the predicted emotion, confidence score, probability chart, emoji, and playlist link.

### Example 4: Serve TensorFlow.js Artifacts

```bash
npx http-server submission/tfjs_model
```

Use `model.json` from the served directory in a TensorFlow.js application.

## Performance & Metrics
- **Train accuracy:** 91.86%
- **Validation accuracy:** 86.72%
- **Test accuracy:** 85.94%
- **Input size:** `48x48` grayscale image
- **Classes:** 7 FER2013 emotion categories
- **Artifacts:** SavedModel, TFLite, and TensorFlow.js exports are included/generated by the pipeline

These metrics are from the existing project documentation and training artifacts. Re-running training may produce slightly different results depending on TensorFlow version, hardware, and random seed behavior.

## Project Statistics
- **Python source files:** 10 scripts in `submission/src`
- **Python source lines:** approximately 1,615 lines
- **Pipeline stages:** dataset preparation, resplitting, training, evaluation, export, CLI inference, Streamlit inference
- **Generated assets:** training history, plots, SavedModel files, TFLite model, TensorFlow.js shards

## Technical Highlights
- **Transfer learning:** Combines a grayscale CNN front-end with an ImageNet-pretrained MobileNetV2 backbone.
- **Two-stage training:** Supports frozen-backbone training followed by partial backbone fine-tuning.
- **Reproducible data flow:** Centralized dataset loading, normalization, augmentation, and label mapping in `utils.py`.
- **Deployment readiness:** Exports the same trained model to server, mobile/edge, and web-compatible formats.
- **User-facing demo:** Streamlit app translates model output into confidence scores, visual probabilities, emoji labels, and playlist recommendations.

## Lessons Learned
Building this project reinforces the importance of treating model development as a full pipeline, not only a notebook experiment. Dataset preparation, evaluation reproducibility, export compatibility, and inference UX all require separate engineering decisions.

The most important technical challenge is adapting small grayscale FER2013 images to a transfer-learning workflow designed for RGB ImageNet inputs. This project handles that by using convolutional preprocessing, channel projection, resizing, and MobileNetV2 fine-tuning. If extended further, the next improvement would be a more robust experiment-tracking setup and a cleaner production packaging layer around model serving.

## Future Improvements
- Add automated tests for preprocessing, label mapping, and inference scripts.
- Add experiment tracking for hyperparameters, metrics, and model versions.
- Add GitHub Actions for linting and smoke tests.
- Add Docker support for reproducible local execution.
- Benchmark latency and model size across SavedModel, TFLite, and TFJS exports.
- Improve fairness and robustness evaluation across different face conditions.

## Author
Luthfi Mirza Darsono
- GitHub: add your GitHub profile link
- LinkedIn: add your LinkedIn profile link

## GitHub Repository Suggestion
- **Repo name:** `fer2013-emotion-classifier`
- **Repo description:** End-to-end TensorFlow pipeline for facial emotion classification with training, evaluation, export, and Streamlit inference.
- **Topics/tags:** `tensorflow`, `keras`, `computer-vision`, `image-classification`, `facial-emotion-recognition`, `streamlit`, `tflite`

## LinkedIn/Portfolio Description
Built an end-to-end TensorFlow FER2013 emotion classifier with evaluation plots, TFLite/TFJS export, and Streamlit demo.
