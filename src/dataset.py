import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
from feature_extraction import extract_log_mel, extract_mfcc, extract_gfcc

class COVIDAudioDataset(Dataset):
    def __init__(self, csv_file, config, augment_pipeline=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.config = config
        self.augment = augment_pipeline
        self.is_train = is_train
        self.feature_type = config['features']['type']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(row['file_path'], sr=self.config['features']['sample_rate'])

        if self.is_train and self.augment:
            audio = self.augment(samples=audio, sample_rate=self.config['features']['sample_rate'])

        if self.feature_type == 'log_mel':
            features = extract_log_mel(audio, sr=self.config['features']['sample_rate'])
        elif self.feature_type == 'mfcc':
            features = extract_mfcc(audio, sr=self.config['features']['sample_rate'])
        elif self.feature_type == 'gfcc':
            features = extract_gfcc(audio, sr=self.config['features']['sample_rate'])

        label = row['label']
        return features, torch.tensor(label, dtype=torch.float32)