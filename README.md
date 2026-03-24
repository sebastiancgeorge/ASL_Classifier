# ASL Classifier

A machine learning project that translates American Sign Language (ASL) hand signals into text using image classification and hand tracking.

## 📋 Project Overview

This project uses computer vision and machine learning to recognize and classify ASL hand gestures. It leverages a hand tracking module for data preprocessing and a TensorFlow-based model built with Google's Teachable Machine for accurate gesture classification.

## 🎯 Features

- **Hand Signal Recognition**: Accurately detects and classifies ASL hand signals
- **Real-time Prediction**: Processes live video input to translate hand gestures
- **Data Preprocessing**: Automated hand tracking and data preprocessing pipeline
- **Model Training**: Built with Teachable Machine for easy model customization
- **User-Friendly Testing**: Simple test script for model validation

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python libraries (see requirements.txt)


## 📊 Workflow

### 1. Data Collection & Preprocessing

Run the data collection script to gather and preprocess hand signal images:

```bash
python DataCollection.py
```

This script will:
- Capture hand gestures using your webcam
- Track hand movements using the hand tracking module
- Preprocess and save the data for model training

### 2. Model Training

1. Download or collect your training data
2. Build your image classification model using [Google's Teachable Machine](https://teachablemachine.withgoogle.com/)
3. Export the trained model to your project directory

### 3. Testing the Model

Test your trained model with the test script:

```bash
python test.py
```

This will:
- Load your trained model
- Process real-time video input
- Display predicted ASL hand signals

## 📁 Project Structure

```
ASL_Classifier/
├── DataCollection.py    # Data collection and preprocessing script
├── test.py             # Model testing script
├── README.md           # This file
└── [model files]       # Trained model files
```

## 📥 Sample Dataset

Sample training data is available at:
[Google Drive Folder](https://drive.google.com/drive/folders/1ZvZ3E3HPDsdUaC8sLdBlhd0dmcmh6zTt?usp=drive_link)

## 🛠️ Technologies Used

- **Language**: Python
- **Libraries**: TensorFlow, OpenCV, MediaPipe (for hand tracking)
- **Model Training**: Google Teachable Machine
- **Image Classification**: Convolutional Neural Networks (CNN)

## 📝 Usage Tips

- Ensure good lighting when collecting data for better hand detection
- Collect diverse samples for each gesture to improve model accuracy
- Use consistent hand positioning and distance from the camera
- Test the model in similar conditions to your training environment

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📄 License

This project is open source and available for educational and personal use.

## 📧 Contact

For questions or suggestions, feel free to reach out via GitHub issues.

---

**Note**: This is an educational project for ASL gesture recognition. For production use, consider using more robust sign language datasets and models.
