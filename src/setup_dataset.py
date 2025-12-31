# # setup_dataset.py
# # Run this script once to organize the Coswara dataset and create train/val/test splits

# import os
# import pandas as pd
# import shutil
# from sklearn.model_selection import train_test_split
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description="Organize Coswara dataset")
#     parser.add_argument("--coswara_path", type=str, required=True,
#                         help="Path to the extracted Coswara-Data folder (the one containing date folders like 20200413 etc.)")
#     parser.add_argument("--output_dir", type=str, default="data/clean_audio",
#                         help="Where to save organized positive/negative folders")
#     parser.add_argument("--metadata_file", type=str, default=None,
#                         help="Optional: direct path to combined_data.csv if you have it")
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     coswara_root = args.coswara_path.rstrip("/")
#     output_pos = os.path.join(args.output_dir, "positive")
#     output_neg = os.path.join(args.output_dir, "negative")
#     os.makedirs(output_pos, exist_ok=True)
#     os.makedirs(output_neg, exist_ok=True)

#     # List of audio types we want to use (these are the most useful for COVID detection)
#     target_audio_types = [
#         'cough-shallow.wav',
#         'cough-heavy.wav',
#         'breathing-deep.wav',
#         'breathing-shallow.wav',
#         'vowel-a.wav',
#         'vowel-e.wav',
#         'vowel-o.wav'
#     ]

#     # Step 1: Find metadata file
#     if args.metadata_file and os.path.exists(args.metadata_file):
#         metadata_path = args.metadata_file
#     else:
#         # Common locations in the repo
#         possible_paths = [
#             os.path.join(coswara_root, "combined_data.csv"),
#             os.path.join(coswara_root, "Coswara_metadata.csv"),
#             "combined_data.csv"  # if you're running from project root
#         ]
#         metadata_path = None
#         for p in possible_paths:
#             if os.path.exists(p):
#                 metadata_path = p
#                 break
#         if metadata_path is None:
#             raise FileNotFoundError("Could not find metadata CSV. Please provide --metadata_file")

#     print(f"Loading metadata from: {metadata_path}")
#     metadata = pd.read_csv(metadata_path)

#     # Clean covid_status: map to binary label
#     def get_label(status):
#         if pd.isna(status):
#             return None
#         status = str(status).lower()
#         if "positive" in status or "resp_illness_not_identified" in status:
#             return "positive"
#         elif "healthy" in status or "no_resp_illness" in status:
#             return "negative"
#         else:
#             return None  # ambiguous or recovered, etc.

#     metadata['label'] = metadata['covid_status'].apply(get_label)
#     metadata = metadata[metadata['label'].notnull()]  # drop ambiguous

#     print(f"Found {len(metadata)} participants with clear labels")
#     print(metadata['label'].value_counts())

#     # Step 2: Copy audio files to positive/negative folders
#     copied_count = 0
#     for _, row in metadata.iterrows():
#         participant_id = row['id']  # usually the folder name
#         label = row['label']

#         # Participant folder can be directly under root or inside date folders
#         possible_paths = [
#             os.path.join(coswara_root, participant_id),
#         ]
#         # Also check inside date subfolders
#         for date_folder in os.listdir(coswara_root):
#             date_path = os.path.join(coswara_root, date_folder)
#             if os.path.isdir(date_path):
#                 participant_path = os.path.join(date_path, participant_id)
#                 if os.path.isdir(participant_path):
#                     possible_paths.append(participant_path)

#         participant_folder = None
#         for p in possible_paths:
#             if os.path.isdir(p):
#                 participant_folder = p
#                 break

#         if participant_folder is None:
#             # print(f"Warning: Folder not found for participant {participant_id}")
#             continue

#         for audio_filename in target_audio_types:
#             src_path = os.path.join(participant_folder, audio_filename)
#             if os.path.exists(src_path):
#                 dest_filename = f"{participant_id}_{audio_filename}"
#                 dest_path = os.path.join(args.output_dir, label, dest_filename)
#                 if not os.path.exists(dest_path):  # avoid duplicates
#                     shutil.copy(src_path, dest_path)
#                     copied_count += 1

#     print(f"Copied {copied_count} audio files to {args.output_dir}")

#     # Step 3: Create train/val/test CSV splits
#     print("Creating data splits...")
#     all_files = []
#     labels = []

#     for label in ["positive", "negative"]:
#         folder = os.path.join(args.output_dir, label)
#         num_label = 1 if label == "positive" else 0
#         for f in os.listdir(folder):
#             if f.endswith(".wav"):
#                 all_files.append(os.path.join(folder, f))
#                 labels.append(num_label)

#     df = pd.DataFrame({"file_path": all_files, "label": labels})

#     # Stratified split to keep class balance
#     train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
#     val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

#     splits_dir = "data/splits"
#     os.makedirs(splits_dir, exist_ok=True)

#     train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
#     val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
#     test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

#     print(f"Splits created:")
#     print(f"Train: {len(train_df)} samples")
#     print(f"Val:   {len(val_df)} samples")
#     print(f"Test:  {len(test_df)} samples")
#     print("\nSetup complete! You can now run training with:")
#     print("python src/train.py")

# if __name__ == "__main__":
#     main()

# setup_dataset.py
# Fixed version - properly searches inside date folders

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Organize Coswara dataset")
    parser.add_argument("--coswara_path", type=str, required=True,
                        help="Path to folder containing date folders (e.g., data/raw)")
    parser.add_argument("--output_dir", type=str, default="data/clean_audio",
                        help="Where to save organized positive/negative folders")
    return parser.parse_args()

def main():
    args = parse_args()
    
    coswara_root = args.coswara_path.rstrip(os.sep)
    output_pos = os.path.join(args.output_dir, "positive")
    output_neg = os.path.join(args.output_dir, "negative")
    os.makedirs(output_pos, exist_ok=True)
    os.makedirs(output_neg, exist_ok=True)

    metadata_path = os.path.join(coswara_root, "combined_data.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"combined_data.csv not found in {coswara_root}")

    print(f"Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    # Binary labeling (common in papers)
    def get_label(status):
        if pd.isna(status):
            return None
        status = str(status).lower()
        if "positive" in status or "recovered" in status:
            return "positive"
        elif "healthy" in status or "no_resp_illness" in status:
            return "negative"
        else:
            return None

    metadata['label'] = metadata['covid_status'].apply(get_label)
    metadata = metadata[metadata['label'].notnull()]
    print(f"Found {len(metadata)} participants with clear labels")
    print(metadata['label'].value_counts())

    # Only copy cough-heavy.wav (best for CPCE)
    target_audio = 'cough-heavy.wav'

    copied_count = 0
    found_participants = 0

    for _, row in metadata.iterrows():
        participant_id = row['id']
        label = row['label']

        participant_folder = None

        # Search inside all date folders
        for date_folder_name in os.listdir(coswara_root):
            date_path = os.path.join(coswara_root, date_folder_name)
            if not os.path.isdir(date_path):
                continue
            potential_path = os.path.join(date_path, participant_id)
            if os.path.isdir(potential_path):
                participant_folder = potential_path
                break

        if participant_folder is None:
            # print(f"Warning: Not found - {participant_id}")
            continue

        found_participants += 1
        src_path = os.path.join(participant_folder, target_audio)
        if os.path.exists(src_path):
            dest_filename = f"{participant_id}_{target_audio}"
            dest_path = os.path.join(args.output_dir, label, dest_filename)
            shutil.copy(src_path, dest_path)
            copied_count += 1

    print(f"Found {found_participants} participant folders")
    print(f"Copied {copied_count} {target_audio} files to {args.output_dir}")

    if copied_count == 0:
        raise ValueError("No audio files copied! Check if cough-heavy.wav exists in participant folders.")

    # Create splits
    all_files = []
    labels_list = []

    for label in ["positive", "negative"]:
        folder = os.path.join(args.output_dir, label)
        num_label = 1 if label == "positive" else 0
        for f in os.listdir(folder):
            if f.endswith(".wav"):
                all_files.append(os.path.join(folder, f))
                labels_list.append(num_label)

    df = pd.DataFrame({"file_path": all_files, "label": labels_list})

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    splits_dir = os.path.join("data", "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    print(f"\nSplits created in {splits_dir}:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    print(f"Class balance (positive): Train ~{train_df['label'].mean():.2%}, Val ~{val_df['label'].mean():.2%}")

if __name__ == "__main__":
    main()