import torch
from torch_geometric.data import DataLoader, Data
import pickle as pkl
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from utils.CGNN import ECGCGNN
from utils.RGNN import ECGRGNN
from utils.attCGNN import ECGattCGNN

# Settings
model_type = 'CGNN'  # 'CGNN' or 'RGNN' or 'attCGNN'
model_path = f'misc/models/{model_type}/best_model.pth'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
with open("misc/dataset/filtered_data.pkl", "rb") as f:
    ecg_signals, labels = pkl.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(['Other', 'Myocardial infarction'])

def normalize_signal(signal):
    return (signal - np.mean(signal, axis=1, keepdims=True)) / (np.std(signal, axis=1, keepdims=True) + 1e-6)

def create_subject_graph(signal: np.ndarray, label_str: str):
    y = torch.tensor([label_encoder.transform([label_str])[0]], dtype=torch.long)
    edge_index = torch.combinations(torch.arange(15), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=torch.tensor(signal, dtype=torch.float), edge_index=edge_index, y=y)

# Prepare data
graph_list = [create_subject_graph(normalize_signal(sig), lbl) for sig, lbl in zip(ecg_signals, labels)]

# Same split as before
from sklearn.model_selection import train_test_split
_, val_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
val_loader = DataLoader(val_graphs, batch_size=1)

# Initialize model
if model_type == 'CGNN':
    model = ECGCGNN(cnn_output_dim=64, gnn_hidden_dim=64, num_classes=2).to(device)
elif model_type == 'RGNN':
    model = ECGRGNN(rnn_output_dim=64, gnn_hidden_dim=64, num_classes=2).to(device)
elif model_type == 'attCGNN':
    model = ECGattCGNN(cnn_output_dim=64, gnn_hidden_dim=64, num_classes=2, num_heads=4).to(device)
else:
    raise ValueError("Invalid model type.")

# Load best model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"Validation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")