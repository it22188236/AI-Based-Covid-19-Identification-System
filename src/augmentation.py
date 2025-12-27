from audiomentations import Compose, AddBackgroundNoise, PitchShift, TimeStretch
import os

def get_augmentation_pipeline(noises_dir, config):
    return Compose([
        AddBackgroundNoise(
            sounds_path=noises_dir,
            min_snr_in_db=config['augmentation']['min_snr_db'],
            max_snr_in_db=config['augmentation']['max_snr_db'],
            p=config['augmentation']['apply_prob']
        ),
        PitchShift(
            min_semitones=config['augmentation']['pitch_shift_semitones'][0],
            max_semitones=config['augmentation']['pitch_shift_semitones'][1],
            p=0.5
        ),
        TimeStretch(
            min_rate=config['augmentation']['time_stretch_rate'][0],
            max_rate=config['augmentation']['time_stretch_rate'][1],
            p=0.5
        )
    ])