import pandas as pd
import numpy as np
import os

#INFOS
REQUIRED_FILES = [
    "Wrist_Flexion.csv",
    "Wrist_Extension.csv",
    "Hand_Open.csv",
    "Hand_Close.csv",
    "No_Motion.csv"
]
MIN_SAMPLES = 500

def validate_subject_folder(folder_path):

    print(f"\nChecking folder: {folder_path}")
    if not os.path.exists(folder_path):
        return "ERROR: Folder does not exist"

    for file in REQUIRED_FILES:
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(file_path):
            print(f"ERROR: Missing file {file}")
            continue
        if os.path.getsize(file_path) == 0:
            print(f"ERROR: {file} is empty")
            continue
        result = validate_raw_emg_file(file_path)
        print(f"{file}: {result}")


def validate_raw_emg_file(path):
    print(f"\nValidating {path}")
    try:
        df = pd.read_csv(path, header=None)
        # 1. Empty check
        if df.shape[0] == 0:
            return "ERROR: Empty file"
        # 2. Single column check
        if df.shape[1] != 1:
            return f"ERROR: Expected 1 column, got {df.shape[1]}"
        signal = df.iloc[:, 0]
        # 3. Numeric check
        if not np.issubdtype(signal.dtype, np.number):
            return "ERROR: Non-numeric values detected"
        # 4. NaN check
        if df.isna().any().any():
            return "ERROR: NaN values detected"
        # 5. Minimum length
        if df.shape[0] < MIN_SAMPLES:
            return f"ERROR: Too few samples (<{MIN_SAMPLES})"
        # 6. Flatline detection
        if signal.std() < 1e-8:
            return "ERROR: Signal appears flat"

        max_abs = np.abs(signal).max()
        if max_abs > 1:
            return "ERROR: Signal amplitude too large (>1 V)"
        if max_abs < 1e-8:
            return "ERROR: Signal amplitude extremely small"
    except Exception as e:
        return f"ERROR: {str(e)}"

    return "OK"


if __name__ == "__main__":
    folder = "./data"
    if not os.path.exists(folder):
        print(f"ERROR: {folder} does not exist")
        exit(1)
    else:
        for subject in os.listdir(folder):
            validate_subject_folder(os.path.join(folder, subject))

"""
au cas où on aurait plus que des folder S0n avec (n allant de 1 à inf) 
et des data.csv dans le folder data 

for subject in os.listdir(folder):
    subject_path = os.path.join(folder, subject)

    if not os.path.isdir(subject_path):
        continue

    validate_subject_folder(subject_path)
    
"""