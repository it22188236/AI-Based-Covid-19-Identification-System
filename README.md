# COVID-19 Lung Segmentation & Analysis System

![COVID-19 Analysis](https://img.shields.io/badge/COVID-19-Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview
A deep learning-based system for automated COVID-19 detection through chest X-ray analysis. This project implements lung segmentation using U-Net architecture followed by COVID-19 classification for medical image analysis.

## 🎯 Key Features
- **Lung Segmentation**: Automatic detection and segmentation of lung regions from chest X-rays
- **COVID-19 Classification**: Rule-based analysis for demonstration purposes
- **Interactive GUI**: User-friendly interface for image upload and analysis
- **Jupyter Notebook**: Interactive demo with real-time results
- **Report Generation**: Automated PDF/image reports with findings


## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/COVID_Lung_Segmentation.git
cd COVID_Lung_Segmentation

# 2. Create virtual environment (optional but recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install tensorflow opencv-python numpy matplotlib pillow



🖥️ Demo Application
How to Use
Run python scripts/demo_app.py

Click "Upload Image" button

Select a chest X-ray image

Click "Analyze" button

View results and download report



📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
MIT License

Copyright (c) 2024 COVID Lung Segmentation Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.