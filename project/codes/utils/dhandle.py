# This file contains functions to load data.
import os
import wfdb
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.processing import preproc
from utils.variables import fs_old, fc, fs, dataset_path

def find_paths(dataset_path):
    paths = []
    for folder in sorted(os.listdir(dataset_path)):
        data_folder = os.path.join(dataset_path, folder)
        if os.path.isdir(data_folder):
            for file in sorted(os.listdir(data_folder)):
                if file.endswith(".dat"):
                    file, ext = os.path.splitext(file)
                    data_file = os.path.join(dataset_path, folder, file)
                    paths.append(data_file)
    return sorted(paths)


def read_ecg_data(root, channels=None):
    record = wfdb.rdrecord(root, channels=channels)
    ecg_data = record.p_signal.T
    label = record.comments[4].split(':')[1][1:]
    return ecg_data, label


def extract_dataset(paths, channels=None):
    records = []
    labels = []
    for path in tqdm(paths, desc="Reading Records: "):
        ecg_data, label = read_ecg_data(path, channels)
        records.append(ecg_data)
        labels.append(label)

    return records, labels

def load_data():
    if os.path.exists("misc/dataset/data.pkl"):
        print("Dataset already exists. Loading...")
        with open("misc/dataset/data.pkl", "rb") as f:
            records, labels = pickle.load(f)
    else:
        print("Dataset not found. Creating...")

        # Prepare data
        paths = find_paths(dataset_path)
        records, labels = extract_dataset(paths)

        # Creating ground truth by an aggressive low-pass filtering
        records = Parallel(n_jobs=-1)(
            delayed(preproc)(record, fc, fs, fs_old, order=64, Q_factor=30, freq=50)
            for record in tqdm(records, desc="Preprocessing ECG signals: ")
        )
        # Save the dataset
        with open("misc/dataset/data.pkl", "wb") as f:
            pickle.dump((records, labels), f)

    return records, labels