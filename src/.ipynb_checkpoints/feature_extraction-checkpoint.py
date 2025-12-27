import torch
import torchaudio.transforms as T
import librosa
import numpy as np

def extract_log_mel(audio, sr=22050, n_mels=128):
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels)
    amp_to_db = T.AmplitudeToDB()
    spec = mel_transform(torch.tensor(audio).unsqueeze(0))
    log_mel = amp_to_db(spec)
    return log_mel.squeeze(0)  # (n_mels, time)

def extract_mfcc(audio, sr=22050, n_mfcc=40):
    mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc)
    mfcc = mfcc_transform(torch.tensor(audio).unsqueeze(0))
    return mfcc.squeeze(0)

# Simple GFCC approximation using gammatone + MFCC
def extract_gfcc(audio, sr=22050, n_mfcc=40):
    # You can replace with proper gammatone library if needed
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmin=20)
    gfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    return torch.tensor(gfcc)