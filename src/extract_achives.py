import os
import tarfile

COSWARA_ROOT = "E:\SLIIT\Year_4\Semester_1\Research Project\RP-Project\coswara_dataset"  # change if needed


def is_date_folder(name):
    return name.isdigit() and len(name) == 8


def merge_split_archives(folder_path, base_name):
    """
    Merge .tar.gz.aa, .ab, .ac ... into a single .tar.gz
    """
    parts = sorted(
        f for f in os.listdir(folder_path)
        if f.startswith(base_name) and f.endswith(tuple(
            [".aa", ".ab", ".ac", ".ad", ".ae", ".af", ".ag", ".ah", ".ai", ".aj"]
        ))
    )

    if not parts:
        return None

    output_tar = os.path.join(folder_path, base_name)
    if os.path.exists(output_tar):
        print(f"✔ {base_name} already merged")
        return output_tar

    print(f"🔗 Merging {len(parts)} parts → {base_name}")

    with open(output_tar, "wb") as outfile:
        for part in parts:
            with open(os.path.join(folder_path, part), "rb") as infile:
                outfile.write(infile.read())

    return output_tar


def extract_tar(tar_path, extract_to):
    print(f"📦 Extracting {os.path.basename(tar_path)}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def already_extracted(folder_path):
    return any(
        os.path.isdir(os.path.join(folder_path, d))
        for d in os.listdir(folder_path)
        if not d.endswith(".tar.gz") and not d.endswith(".csv")
    )


def main():
    for folder in os.listdir(COSWARA_ROOT):
        folder_path = os.path.join(COSWARA_ROOT, folder)

        if not os.path.isdir(folder_path):
            continue

        if not is_date_folder(folder):
            continue

        print(f"\n📁 Processing folder: {folder}")

        if already_extracted(folder_path):
            print("✔ Already extracted, skipping")
            continue

        base_name = f"{folder}.tar.gz"
        tar_path = merge_split_archives(folder_path, base_name)

        if tar_path and os.path.exists(tar_path):
            extract_tar(tar_path, folder_path)
            # Optional: delete merged tar to save space
            # os.remove(tar_path)
        else:
            print("⚠ No split archive found")


if __name__ == "__main__":
    main()
