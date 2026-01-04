# AI-Based COVID-19 Identification System

An **explainable** deep learning system for detecting COVID-19 from chest X-rays using **ResNet50** and **Grad-CAM**.

## Key Features
- 3-class classification: **COVID-19**, **Normal**, **Viral Pneumonia**
- Achieved **~93% validation accuracy**
- **Grad-CAM heatmaps** showing clinically relevant model attention on lung regions
- Full end-to-end reproducible pipeline

## Results
- Training history plot: [`results/plots/training_history.png`](results/plots/training_history.png)
- Example Grad-CAM visualizations: [`results/heatmaps/`](results/heatmaps/)

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.train

# Generate Grad-CAM heatmaps
python -m src.evaluate

# Explore interactively
jupyter notebook
# Then open notebooks/exploratory.ipynb
