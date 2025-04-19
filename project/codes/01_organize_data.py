import pickle as pkl
from collections import Counter
from utils.variables import dataset_path
from utils.dhandle import extract_dataset, find_paths

paths = find_paths(dataset_path)
records, labels = extract_dataset(paths)
freq = Counter(labels)
print(freq)
# {'Myocardial infarction': 368, 'Healthy control': 80, 'n/a': 27, 'Cardiomyopathy': 17, ...}

# Only keep records with specific labels
target_conditions = {'Myocardial infarction', 'Healthy Control'}
filtered_records = []
filtered_labels = []

for label, record in zip(labels, records):
    if label in target_conditions:
        filtered_records.append(record)
        filtered_labels.append(label)

# Save the filtered records and labels
with open("misc/dataset/filtered_data.pkl", "wb") as f:
    pkl.dump((filtered_records, filtered_labels), f)

# # Load the filtered records and labels
# with open("misc/dataset/filtered_data.pkl", "rb") as f:
#     filtered_records, filtered_labels = pkl.load(f)