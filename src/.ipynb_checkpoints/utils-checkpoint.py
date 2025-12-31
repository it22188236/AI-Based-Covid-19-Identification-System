# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# def load_dataset(data_dir, metadata_csv='metadata.csv'):
#     """Load Coswara: assumes WAV in raw/, metadata CSV."""
#     df = pd.read_csv(os.path.join(data_dir, metadata_csv))
#     df = df[df['covid_status'].isin(['positive', 'healthy'])]  # Binary
#     labels = (df['covid_status'] == 'positive').astype(int).values
    
#     features = []
#     for idx, row in df.iterrows():
#         file_path = os.path.join(data_dir, 'raw', row['id'] + '_cough-heavy.wav')  # Adjust filename
#         if os.path.exists(file_path):
#             features.append(file_path)  # Placeholder; extract later
#     return features, labels

# def save_features(features, labels, out_dir):
#     np.savez(os.path.join(out_dir, 'features.npz'), X=features, y=labels)

# def load_splits(processed_dir):
#     data = np.load(os.path.join(processed_dir, 'features.npz'))
#     X, y = data['X'], data['y']
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
#     return X_train, y_train, X_val, y_val, X_test, y_test

# src/utils.py

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split


# def load_dataset(data_dir):
#     """
#     Load all existing cough-heavy.wav file paths and corresponding labels.
    
#     Safe for current Coswara structure: no reliance on non-existent columns.
#     """
#     raw_dir = os.path.join(data_dir, 'raw')
#     metadata_path = os.path.join(raw_dir, 'combined_data.csv')
    
#     if not os.path.exists(metadata_path):
#         raise FileNotFoundError(
#             f"\n⚠️ combined_data.csv not found at {metadata_path}\n"
#             "Copy it from the original Coswara-Data repo into data/raw/"
#         )
    
#     print("Loading metadata from combined_data.csv...")
#     df = pd.read_csv(metadata_path)
    
#     # Print available columns for debugging (remove later if desired)
#     print(f"Available columns: {list(df.columns)}")
    
#     # Filter only positive/healthy for binary classification
#     initial_count = len(df)
#     df = df[df['covid_status'].isin(['positive', 'healthy'])]
#     print(f"Filtered to {len(df)} samples with positive/healthy status (from {initial_count} total)")
    
#     audio_paths = []
#     labels = []
#     missing_count = 0
    
#     base_dir = os.path.join(data_dir, 'raw')
    
#     for _, row in df.iterrows():
#         participant_id = row['id']
#         found = False
        
#         # Search across all date folders (e.g., 20200413, etc.)
#         for date_folder in os.listdir(base_dir):
#             date_path = os.path.join(base_dir, date_folder)
#             if os.path.isdir(date_path):
#                 file_path = os.path.join(date_path, participant_id, 'cough-heavy.wav')
#                 if os.path.exists(file_path):
#                     audio_paths.append(file_path)
#                     labels.append(1 if row['covid_status'] == 'positive' else 0)
#                     found = True
#                     break
        
#         if not found:
#             missing_count += 1
    
#     print(f"Found {len(audio_paths)} valid cough-heavy.wav files")
#     if missing_count > 0:
#         print(f"Skipped {missing_count} participants (no cough-heavy.wav found)")
    
#     if len(audio_paths) == 0:
#         raise ValueError("No cough-heavy.wav files found! Check data structure.")
    
#     return audio_paths, np.array(labels)


# def save_features(features, labels, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     save_path = os.path.join(out_dir, 'cpce_features.npz')
#     np.savez(save_path, X=features, y=labels)
#     print(f"CPCE features saved to {save_path} ({len(features)} samples)")


# def load_splits(processed_dir):
#     features_path = os.path.join(processed_dir, 'cpce_features.npz')
    
#     if not os.path.exists(features_path):
#         raise FileNotFoundError(f"No features at {features_path}. Run feature extraction first.")
    
#     data = np.load(features_path)
#     X = data['X']
#     y = data['y']
    
#     print(f"Loaded {len(X)} CPCE features")
    
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
#     )
    
#     print(f"Splits → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
#     return X_train, y_train, X_val, y_val, X_test, y_test

# src/utils.py

# import os
# import json
# import numpy as np
# from sklearn.model_selection import train_test_split


# def load_dataset(data_dir):
#     """
#     Load cough-heavy.wav paths and labels directly from per-participant metadata.json files.
    
#     Works perfectly with your current structure:
#     data/raw/<date_folder>/<participant_id>/cough-heavy.wav
#     data/raw/<date_folder>/<participant_id>/metadata.json
#     """
#     raw_dir = os.path.join(data_dir, 'raw')
    
#     if not os.path.exists(raw_dir):
#         raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
#     audio_paths = []
#     labels = []
    
#     valid_count = 0
#     skipped_no_audio = 0
#     skipped_no_metadata = 0
#     skipped_invalid_status = 0
    
#     print("Scanning data/raw/ for cough-heavy.wav and metadata.json...")
    
#     # Traverse date folders
#     for date_folder in sorted(os.listdir(raw_dir)):
#         date_path = os.path.join(raw_dir, date_folder)
#         if not os.path.isdir(date_path):
#             continue  # Skip files like combined_data.csv if present
        
#         # Traverse participant folders
#         for participant_folder in os.listdir(date_path):
#             participant_path = os.path.join(date_path, participant_folder)
#             if not os.path.isdir(participant_path):
#                 continue
            
#             audio_path = os.path.join(participant_path, 'cough-heavy.wav')
#             metadata_path = os.path.join(participant_path, 'metadata.json')
            
#             if not os.path.exists(audio_path):
#                 skipped_no_audio += 1
#                 continue
            
#             if not os.path.exists(metadata_path):
#                 skipped_no_metadata += 1
#                 continue
            
#             try:
#                 with open(metadata_path, 'r', encoding='utf-8') as f:
#                     metadata = json.load(f)
                
#                 # Key is 'covid_status' based on official Coswara usage
#                 covid_status = metadata.get('covid_status', '').strip().lower()
                
#                 if covid_status in ['positive', 'healthy']:
#                     audio_paths.append(audio_path)
#                     labels.append(1 if covid_status == 'positive' else 0)
#                     valid_count += 1
#                 else:
#                     skipped_invalid_status += 1
                    
#             except json.JSONDecodeError:
#                 print(f"Warning: Invalid JSON in {metadata_path}")
#                 skipped_no_metadata += 1
    
#     print(f"\n=== Dataset Loading Summary ===")
#     print(f"Valid samples (cough-heavy.wav + positive/healthy status): {valid_count}")
#     print(f"Skipped (no cough-heavy.wav): {skipped_no_audio}")
#     print(f"Skipped (no/invalid metadata.json): {skipped_no_metadata}")
#     print(f"Skipped (other covid_status like 'resp_illness_not_identified'): {skipped_invalid_status}")
    
#     if valid_count == 0:
#         raise ValueError("No valid samples found! Check if metadata.json contains 'covid_status' key with 'positive' or 'healthy'.")
    
#     return audio_paths, np.array(labels)


# # Rest of the functions remain the same
# def save_features(features, labels, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     save_path = os.path.join(out_dir, 'cpce_features.npz')
#     np.savez(save_path, X=features, y=labels)
#     print(f"\nCPCE features saved: {save_path} ({len(features)} samples)")


# def load_splits(processed_dir):
#     features_path = os.path.join(processed_dir, 'cpce_features.npz')
#     if not os.path.exists(features_path):
#         raise FileNotFoundError(f"Features not found: {features_path}. Run extraction first.")
    
#     data = np.load(features_path)
#     X = data['X']
#     y = data['y']
    
#     print(f"Loaded {len(X)} features")
    
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
#     print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
#     return X_train, y_train, X_val, y_val, X_test, y_test


# src/utils.py

# import os
# import json
# import numpy as np
# from sklearn.model_selection import train_test_split


# def load_dataset(data_dir):
#     raw_dir = os.path.join(data_dir, 'raw')
    
#     if not os.path.exists(raw_dir):
#         raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
#     audio_paths = []
#     labels = []  # 1 = COVID-positive (or recovered), 0 = healthy/non-COVID
    
#     valid_count = 0
#     skipped_no_audio = 0
#     skipped_no_metadata = 0
#     status_counts = {}  # For debugging: count each covid_status
    
#     print("Scanning data/raw/ for cough-heavy.wav and metadata.json...")
    
#     for date_folder in sorted(os.listdir(raw_dir)):
#         date_path = os.path.join(raw_dir, date_folder)
#         if not os.path.isdir(date_path):
#             continue
        
#         for participant_folder in os.listdir(date_path):
#             participant_path = os.path.join(date_path, participant_folder)
#             if not os.path.isdir(participant_path):
#                 continue
            
#             audio_path = os.path.join(participant_path, 'cough-heavy.wav')
#             metadata_path = os.path.join(participant_path, 'metadata.json')
            
#             if not os.path.exists(audio_path):
#                 skipped_no_audio += 1
#                 continue
            
#             if not os.path.exists(metadata_path):
#                 skipped_no_metadata += 1
#                 continue
            
#             try:
#                 with open(metadata_path, 'r', encoding='utf-8') as f:
#                     metadata = json.load(f)
                
#                 status = metadata.get('covid_status', '').strip().lower()
                
#                 # Count for debugging
#                 status_counts[status] = status_counts.get(status, 0) + 1
                
#                 # Broader binary classification used in many papers
#                 if status in ['positive', 'recovered']:
#                     audio_paths.append(audio_path)
#                     labels.append(1)  # Positive
#                     valid_count += 1
#                 elif status in ['healthy', 'no_resp_illness_exposed', 'resp_illness_not_identified']:
#                     audio_paths.append(audio_path)
#                     labels.append(0)  # Negative/Non-COVID
#                     valid_count += 1
                    
#             except json.JSONDecodeError as e:
#                 print(f"Warning: Invalid JSON in {metadata_path}: {e}")
#                 skipped_no_metadata += 1
    
#     print(f"\n=== Dataset Loading Summary ===")
#     print(f"Valid samples found: {valid_count}")
#     print(f"Skipped (no cough-heavy.wav): {skipped_no_audio}")
#     print(f"Skipped (no/invalid metadata.json): {skipped_no_metadata}")
#     print(f"All covid_status values found: {status_counts}")
    
#     if valid_count == 0:
#         raise ValueError(
#             "No valid samples found! Your dataset subset has no 'positive', 'healthy', or common categories.\n"
#             "Try downloading/extracting more date folders from the full Coswara repo."
#         )
    
#     return audio_paths, np.array(labels)


# def save_features(features, labels, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     save_path = os.path.join(out_dir, 'cpce_features.npz')
#     np.savez(save_path, X=features, y=labels)
#     print(f"\nCPCE features saved: {save_path} ({len(features)} samples, {np.sum(labels)} positive)")


# def load_splits(processed_dir):
#     features_path = os.path.join(processed_dir, 'cpce_features.npz')
#     if not os.path.exists(features_path):
#         raise FileNotFoundError(f"Features not found: {features_path}. Run extraction first.")
    
#     data = np.load(features_path)
#     X = data['X']
#     y = data['y']
    
#     print(f"Loaded {len(X)} features ({np.sum(y)} positive, {len(y)-np.sum(y)} negative)")
    
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
#     print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
#     return X_train, y_train, X_val, y_val, X_test, y_test

# src/utils.py

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split


# def load_dataset(data_dir, organized=True):
#     """
#     Load cough-heavy.wav paths and labels.
    
#     If organized=True (recommended): uses data/clean_audio/positive & negative folders
#     created by setup_dataset.py
    
#     If organized=False: falls back to raw traversal (your current method)
#     """
#     if organized:
#         clean_dir = os.path.join(data_dir, "clean_audio")
#         if not os.path.exists(clean_dir):
#             raise FileNotFoundError(
#                 f"Organized data not found at {clean_dir}\n"
#                 "Run setup_dataset.py first to create positive/negative folders."
#             )
        
#         audio_paths = []
#         labels = []
        
#         for label_name, label_val in [("positive", 1), ("negative", 0)]:
#             folder = os.path.join(clean_dir, label_name)
#             if not os.path.exists(folder):
#                 continue
#             for filename in os.listdir(folder):
#                 if filename.endswith("cough-heavy.wav"):
#                     audio_paths.append(os.path.join(folder, filename))
#                     labels.append(label_val)
        
#         print(f"Loaded {len(audio_paths)} organized cough-heavy samples "
#               f"({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
        
#         return audio_paths, np.array(labels)
    
#     else:
#         # Your previous raw traversal code (kept as fallback)
#         # ... (the full traversal code from before)
#         pass  # Omit for brevity — you can keep it if needed


# def save_features(features, labels, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     save_path = os.path.join(out_dir, 'cpce_features.npz')
#     np.savez(save_path, X=features, y=labels)
#     print(f"\nCPCE features saved: {save_path} "
#           f"({len(features)} samples, {np.sum(labels)} positive)")


# def load_splits(processed_dir):
#     features_path = os.path.join(processed_dir, 'cpce_features.npz')
#     if not os.path.exists(features_path):
#         raise FileNotFoundError(f"Features not found: {features_path}. Run extraction first.")
    
#     data = np.load(features_path)
#     X = data['X']
#     y = data['y']
    
#     print(f"Loaded {len(X)} CPCE features ({np.sum(y)} positive)")
    
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
#     print(f"Splits → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
#     return X_train, y_train, X_val, y_val, X_test, y_test

# src/utils.py - Updated for clean_audio + CSV splits structure

import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split  # Not needed anymore but kept for compatibility

def load_dataset(data_dir):
    """
    Load from clean setup_dataset.py structure:
    data/clean_audio/{positive,negative}/*.wav
    data/splits/{train,val,test}.csv
    """
    splits_dir = os.path.join(data_dir, 'splits')
    clean_dir = os.path.join(data_dir, 'clean_audio')
    
    if not os.path.exists(splits_dir):
        raise FileNotFoundError(
            f"Splits not found at {splits_dir}\n"
            "Run: python setup_dataset.py --coswara_path 'data/raw' --output_dir 'data/clean_audio'"
        )
    
    # Load splits
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
    
    print(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Class balance - Train: {train_df['label'].value_counts().to_dict()}")
    
    return {
        'train': (train_df['file_path'].values, train_df['label'].values),
        'val': (val_df['file_path'].values, val_df['label'].values),
        'test': (test_df['file_path'].values, test_df['label'].values)
    }

# def load_split(split_name, data_dir):
#     """Load single split (audio_paths, labels)"""
#     splits_dir = os.path.join(data_dir, 'splits')
#     df = pd.read_csv(os.path.join(splits_dir, f'{split_name}.csv'))
#     return df['file_path'].values, df['label'].values

# def load_split(split_name, data_dir):
#     """Load one split (e.g., 'train')"""
#     splits_dir = os.path.join(data_dir, 'splits')
#     df = pd.read_csv(os.path.join(splits_dir, f'{split_name}.csv'))
#     return df['file_path'].values.tolist(), df['label'].values

def load_splits(processed_dir):
    """
    Load ALL pre-extracted CPCE features (train, val, test) from .npz files
    This is what you need in notebooks 03 and 04!
    """
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"No processed folder at {processed_dir}. Run main.py first!")
    
    pattern = os.path.join(processed_dir, 'cpce_features_*.npz')
    files = glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No cpce_features_*.npz files found in {processed_dir}")
    
    features = {}
    for file in files:
        split_name = os.path.basename(file).replace('cpce_features_', '').replace('.npz', '')
        data = np.load(file)
        features[split_name] = (data['X'], data['y'])
        print(f"Loaded {split_name}: {data['X'].shape[0]} samples")
    
    # Return in order
    return (
        features['train'][0], features['train'][1],
        features['val'][0],   features['val'][1],
        features['test'][0],  features['test'][1]
    )

def save_features(features_dict, data_dir):
    """Save CPCE features by split"""
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    for split_name, (X, y) in features_dict.items():
        np.savez(
            os.path.join(processed_dir, f'cpce_features_{split_name}.npz'),
            X=X, y=y
        )
    print("All CPCE features saved by split")

def load_features(processed_dir='data/processed'):
    import glob
    files = glob.glob(os.path.join(processed_dir, 'cpce_features_*.npz'))
    features = {}
    for f in files:
        split = os.path.basename(f).split('_')[-1].replace('.npz', '')
        data = np.load(f)
        features[split] = (data['X'], data['y'])
    return features