📌Project Overview

This project is an AI-powered system that analyzes cough sounds and classifies them as Normal or Possible Risk. It uses audio feature extraction and deep learning to make predictions.

🚀Features

Upload or record cough audio

AI-based cough classification

Noise-robust audio processing

Simple and user-friendly interface

🧠 Technologies Used

Python

TensorFlow / Keras

Librosa

NumPy

Git & GitHub

project/
│
├── data/
│   ├── clean_cough/
│   ├── mixed_cough/
│
├── model/
│   └── cough_model.h5
│
├── app/
│   ├── ui/
│   └── logic/
│
├── docs/
│   ├── sketches/
│   ├── design_report.pdf
│
└── README.md



⚙️ How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git

2. Install Dependencies
pip install -r requirements.txt

3. Run the Prediction Script
python predict.py

🎯 Model Workflow

Audio input (.wav file)

Feature extraction (MFCC + Gammatone)

🎯Future Improvements

Real-time microphone recording

Mobile application deployment

Cloud-based prediction

Larger dataset for better accuracy

Model prediction

Result display
