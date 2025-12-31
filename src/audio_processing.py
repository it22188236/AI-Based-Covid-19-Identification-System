import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, find_peaks

def load_audio(file_path):
    """Load raw WAV audio."""
    sr, audio = wavfile.read(file_path)
    audio = audio.astype(np.float32) / np.max(np.abs(audio))  # Normalize
    return audio, sr

def segment_phases(audio, sr, plot):
    """Segment cough into 4 phases: inspiration, compression, expulsion, glottis.
    Uses envelope detection and energy thresholding."""
    # Compute analytic signal envelope
    analytic_signal = hilbert(audio)
    envelope = np.abs(analytic_signal)
    
    # Normalize envelope
    envelope /= np.max(envelope)
    
    # Find peaks and thresholds
    peaks, _ = find_peaks(envelope, height=0.1)
    if len(peaks) < 3:
        raise ValueError("Invalid cough audio: too few peaks.")
    
    # Approximate phase boundaries (heuristic; tune thresholds)
    start = 0
    insp_end = peaks[0]  # Inspiration: initial low-energy rise
    comp_end = peaks[1]  # Compression: build-up to peak
    exp_end = peaks[-1]  # Expulsion: main burst
    end = len(audio)     # Glottis: decay after last peak
    
    phases = {
        'inspiration': audio[start:insp_end],
        'compression': audio[insp_end:comp_end],
        'expulsion': audio[comp_end:exp_end],
        'glottis': audio[exp_end:end]
    }
    return phases