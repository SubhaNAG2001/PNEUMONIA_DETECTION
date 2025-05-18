# PNEUMONIA_DETECTION
# Pneumonia Detection from Chest X-rays

A web application that uses deep learning to detect pneumonia from chest X-ray images.

## Overview

This project implements a convolutional neural network (CNN) trained on chest X-ray images to classify them as either normal or showing signs of pneumonia. The model is deployed as a web application using Flask, allowing users to upload X-ray images and receive predictions.

## Features

- Upload chest X-ray images through a user-friendly web interface
- Real-time prediction using a pre-trained deep learning model
- Display of prediction results with the uploaded image
- Responsive design using Tailwind CSS

## Tech Stack

- **Frontend**: HTML, CSS, Tailwind CSS
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Keras preprocessing

## Model Architecture

The pneumonia detection model is a CNN with the following architecture:
- 3 convolutional layers with max pooling
- Dropout layer for regularization
- Dense layers for classification
- Trained on over 5,000 labeled chest X-ray images
- Achieves ~90% accuracy on test data

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. Access the web interface at `http://127.0.0.1:5000/`
2. Upload a chest X-ray image using the provided form
3. Click "Predict" to get the classification result
4. View the prediction (Normal or Pneumonia) along with the uploaded image

## Dataset

The model was trained on the Chest X-Ray Images (Pneumonia) dataset from Kaggle, which contains:
- 5,216 training images
- 16 validation images
- 624 test images
- Two classes: NORMAL and PNEUMONIA

## Author

Subhajit Nag

## License

[MIT License](LICENSE)

## Acknowledgements

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- TensorFlow and Keras for the deep learning framework
- Flask for the web application framework
