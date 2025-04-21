import pickle as pkl
from collections import Counter
from utils.dhandle import load_data

records, labels = load_data()
freq = Counter(labels)
print(freq)
# {'Myocardial infarction': 368, 'Healthy control': 80, 'n/a': 27, 'Cardiomyopathy': 17, ...}

# Only keep records with specific labels
filtered_labels = []

for label, record in zip(labels, records):
    if label == 'Myocardial infarction':
        filtered_labels.append(label)
    else:
        filtered_labels.append('Other')


# Save the filtered records and labels
with open("misc/dataset/filtered_data.pkl", "wb") as f:
    pkl.dump((records, filtered_labels), f)

# # Load the filtered records and labels
# with open("misc/dataset/filtered_data.pkl", "rb") as f:
#     filtered_records, filtered_labels = pkl.load(f)