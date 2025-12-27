# src/evaluate.py
# This script evaluates the trained model on noisy test sets at various SNR levels
# It creates noisy versions of the clean test set, computes metrics (accuracy, precision, recall, F1, AUC)
# and saves/plots results (accuracy vs SNR curve)

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from tqdm import tqdm
import librosa
from audiomentations import AddBackgroundNoise, Compose
from dataset import COVIDAudioDataset  # Reuse feature extraction from training
from model import SimpleCNN
from utils import load_config

# Fixed SNR levels to evaluate (in dB) – common range for real-world testing
SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30]

def add_noise_at_snr(audio, noise_audio, target_snr_db, sr=22050):
    """
    Mix clean audio with noise at exact target SNR.
    Returns noisy audio.
    """
    # Trim or repeat noise to match audio length
    if len(noise_audio) < len(audio):
        noise_audio = np.tile(noise_audio, int(np.ceil(len(audio) / len(noise_audio))))[:len(audio)]
    else:
        start = np.random.randint(0, len(noise_audio) - len(audio))
        noise_audio = noise_audio[start:start + len(audio)]

    # Calculate RMS
    audio_rms = np.sqrt(np.mean(audio**2))
    noise_rms = np.sqrt(np.mean(noise_audio**2))

    if noise_rms == 0:
        return audio  # avoid division by zero

    # Desired noise RMS for target SNR
    desired_noise_rms = audio_rms / (10**(target_snr_db / 20))

    # Scale noise
    scaled_noise = noise_audio * (desired_noise_rms / noise_rms)

    # Mix
    noisy_audio = audio + scaled_noise

    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val

    return noisy_audio

def load_noise_files(noises_dir):
    """Load all noise files from the noises directory (all subfolders)"""
    noise_files = []
    for root, _, files in os.walk(noises_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                noise_files.append(os.path.join(root, f))
    print(f"Loaded {len(noise_files)} noise files for evaluation.")
    return noise_files

def evaluate_on_noisy_test(model, test_df, config, noise_files, device):
    """Main evaluation loop over SNR levels"""
    results = []

    # Use the same feature extraction as during training
    feature_type = config['features']['type']
    sr = config['features']['sample_rate']

    model.eval()

    for snr_db in SNR_LEVELS:
        print(f"\nEvaluating at SNR = {snr_db} dB...")
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"SNR {snr_db}dB"):
                audio, _ = librosa.load(row['file_path'], sr=sr)

                # Randomly pick a noise file
                noise_path = np.random.choice(noise_files)
                noise_audio, _ = librosa.load(noise_path, sr=sr)

                # Add noise at exact SNR
                noisy_audio = add_noise_at_snr(audio, noise_audio, snr_db, sr)

                # Extract features (same as training)
                if feature_type == 'log_mel':
                    from feature_extraction import extract_log_mel
                    features = extract_log_mel(noisy_audio, sr=sr)
                elif feature_type == 'mfcc':
                    from feature_extraction import extract_mfcc
                    features = extract_mfcc(noisy_audio, sr=sr)
                elif feature_type == 'gfcc':
                    from feature_extraction import extract_gfcc
                    features = extract_gfcc(noisy_audio, sr=sr)

                features = features.unsqueeze(0).to(device)  # (1, freq, time)

                output = model(features)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob > 0.5 else 0

                all_probs.append(prob)
                all_preds.append(pred)
                all_labels.append(row['label'])

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)

        results.append({
            'SNR_dB': snr_db,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc
        })

        print(f"SNR {snr_db}dB → Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

    return pd.DataFrame(results)

def plot_results(results_df, save_path="results/plots/accuracy_vs_snr.png"):
    """Plot accuracy and AUC vs SNR"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['SNR_dB'], results_df['Accuracy'], marker='o', label='Accuracy')
    plt.plot(results_df['SNR_dB'], results_df['AUC'], marker='s', label='AUC')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Score')
    plt.title('Model Performance vs Noise Level (SNR)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    config = load_config("../configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = SimpleCNN().to(device)
    model_path = config['training']['model_save_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first!")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    # Load test split
    test_csv = "../data/splits/test.csv"
    test_df = pd.read_csv(test_csv)

    # Load noise files
    noises_dir = os.path.join(config['data']['root'], config['data']['noises_dir'])
    noise_files = load_noise_files(noises_dir)

    # Run evaluation
    results_df = evaluate_on_noisy_test(model, test_df, config, noise_files, device)

    # Save metrics
    metrics_path = "results/metrics/noisy_evaluation_results.csv"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    results_df.to_csv(metrics_path, index=False)
    print(f"Full metrics saved to {metrics_path}")

    # Plot
    plot_results(results_df)

    print("\nEvaluation complete! Compare this curve with a baseline (clean-trained) model to show improvement.")

if __name__ == "__main__":
    main()