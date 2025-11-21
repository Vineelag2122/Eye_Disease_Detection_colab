# Eye Disease Detection (Colab + Flask)

A lightweight Eye Disease Classification system using deep learning (transfer learning with MobileNetV2) and a Flask web application for quick, real-time predictions on eye fundus images.

## Table of Contents
- [Overview](#overview)
- [Supported Classes](#supported-classes)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Model & Training Overview](#model--training-overview)
- [Files / Project Structure](#files--project-structure)
- [How to Run (Colab / Locally)](#how-to-run-colab--locally)
  - [In Google Colab (recommended for training)](#in-google-colab-recommended-for-training)
  - [Locally (Flask prediction app)](#locally-flask-prediction-app)
- [Example Usage](#example-usage)
- [Tips for Improving Accuracy](#tips-for-improving-accuracy)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

## Overview
This project demonstrates building a classifier for common eye conditions using transfer learning. It uses a pretrained MobileNetV2 backbone as a feature extractor, adds a few dense layers on top, and serves predictions through a simple Flask web UI.

## Supported Classes
- Normal
- Cataract
- Diabetic Retinopathy
- Glaucoma

## Dataset
Primary dataset used:
- Kaggle — Eye Diseases Classification: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

Download and prepare the dataset (resize, normalization, split into train/val/test) before training.

## Technologies
- Python (3.7+ recommended)
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- NumPy, Pillow (PIL)
- Matplotlib (for training plots)
- Flask (simple web app for predictions)
- Werkzeug (file handling in Flask)
- OS module, other standard utilities

## Model & Training Overview
- Base model: MobileNetV2 (pretrained on ImageNet) used as a feature extractor
- Head: GlobalAveragePooling2D -> Dense layers -> Dropout -> Softmax output for 4 classes
- Optimizer: Adam (tunable learning rate)
- Loss: Categorical cross-entropy (for one-hot labels) or sparse categorical cross-entropy (for integer labels)
- Regularization: Dropout and data augmentation recommended to reduce overfitting

Notes:
- Save model weights (e.g., model.h5 or SavedModel) after training, so the Flask app can load them for inference.
- Monitor training/validation accuracy and loss; use early stopping or model checkpoints.

## Files / Project Structure (suggested)
This repository may include or should include:
- notebooks/
  - training_colab.ipynb — Colab notebook for training and evaluation
- app.py (or app/main.py) — Flask application to serve predictions
- model/ or saved_models/ — place trained model artifacts here (model.h5, labels.json)
- requirements.txt — Python dependencies
- README.md — this file
- static/, templates/ — frontend files for Flask UI
- utils.py — preprocessing, label mapping, helper functions

Adjust names based on your actual repo layout.

## How to Run (Colab / Locally)

### In Google Colab (recommended for training)
1. Open the provided training notebook (notebooks/training_colab.ipynb) or create one.
2. Mount Google Drive for dataset and model storage:
   - from google.colab import drive
   - drive.mount('/content/drive')
3. Install dependencies (if needed):
   - !pip install -r requirements.txt
4. Prepare data: load images, resize (e.g., 224x224 for MobileNetV2), normalize using MobileNetV2 preprocess_input.
5. Train model, save checkpoints and final model to Drive.

### Locally (Flask prediction app)
1. Create and activate a virtual environment:
   - python -m venv venv
   - source venv/bin/activate  (or venv\Scripts\activate on Windows)
2. Install dependencies:
   - pip install -r requirements.txt
3. Place your trained model file in the model/ or saved_models/ folder.
4. Run the Flask app (example):
   - python app.py
   - By default Flask serves at http://127.0.0.1:5000
5. Use the web UI to upload an image and get a prediction.

Note: The exact Flask entrypoint filename may be app.py, run.py, or similar in your repo—use whichever exists.

## Example Usage
- Training (Colab):
  - Train for N epochs, use ModelCheckpoint to save best weights.
  - Visualize metrics with Matplotlib.
- Inference (Flask):
  - Upload fundus image via the web form, the backend loads the model, preprocesses image, runs model.predict, and returns the predicted class + confidence score.

## Tips for Improving Accuracy
- Use data augmentation (rotation, flips, brightness jitter) to increase dataset variety.
- Fine-tune some last layers of MobileNetV2 (unfreeze a few top layers) after initial training.
- Use class weighting if classes are imbalanced or perform oversampling/undersampling.
- Try larger input size or alternate backbone (EfficientNet, ResNet) if compute allows.
- Use cross-validation for more robust evaluation.

## Contributing
Contributions, PRs, and issue reports are welcome. Suggested ways to help:
- Add a Colab notebook with step-by-step training and evaluation.
- Provide a requirements.txt with exact versions.
- Add unit tests or CI checks.
- Improve Flask UI and error handling.

## License & Contact
Include your chosen license (e.g., MIT) and contact info or GitHub handle:
- Author: Vineelag2122
- Repo: https://github.com/Vineelag2122/Eye_Disease_Detection_colab

If you'd like, I can:
- add a detailed example training Colab notebook,
- generate a sample requirements.txt,
- or update the repo with a ready-to-run Flask example (app.py + templates). Tell me which one you want next and I'll prepare it.
