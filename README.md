# Waste Classification Project

## Project Description

The goal of this project is to build a smart waste classification system that can identify different types of  garbage from images and recommend the correct disposal bin. This helps improve recycling accuracy and support better waste management.

## General Information

This repository contains computer vision experiments, trained models, and a Streamlit inference app for classifying waste into 10 categories. The project uses transfer learning with ResNet50, EfficientNet-B0, and MobileNetV2 to build practical waste classifiers.

## Dataset

The dataset is based on the Kaggle Garbage Classification dataset:

- https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data

The dataset includes images of household waste separated into classes such as paper, cardboard, metal, plastic, glass, clothes, shoes, battery, biological waste, and trash. In this project, glass subcategories are merged into a single `glass` category.

## Project Includes

- `streamlit_app.py` - Main Streamlit application for inference.
- `models/resnet50_waste_classifier_v3.keras` - ResNet50 waste classifier.
- `models/efficientnet_waste_classifier_v5.keras` - EfficientNet-B0 waste classifier.
- `models/mobilenet_waste_classifier_v4.keras` - MobileNetV2 waste classifier.
- `assets/test_images/` - Example images for testing.
- `train_cnn.ipynb` - Notebook for training a custom CNN model.
- `train_mobileNet.ipynb` - Notebook for training and evaluating MobileNetV2.
- `train_effecientNet.ipynb` - Notebook for training and evaluating EfficientNet.
- `train_resnet50.ipynb` - Notebook for training and evaluating ResNet50.

## Technologies

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
- Matplotlib
- OpenCV

## How to Run

1. Activate the Python environment:

```bash
source venv/bin/activate
```

2. Install dependencies if needed:

```bash
pip install streamlit tensorflow pillow numpy matplotlib opencv-python
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Notes

- The app loads pre-trained `.keras` model files from `models/`.
- The project uses these waste categories: `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, and `trash`.
- If prediction results are inconsistent, make sure preprocessing in the app matches the training preprocessing.


