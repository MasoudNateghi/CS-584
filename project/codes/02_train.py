import os
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.CGNN import ECGCGNN
from utils.RGNN import ECGRGNN

model_type = 'RGNN'  # 'CGNN' or 'RGNN'
model_dir = f'misc/models/{model_type}'
os.makedirs(model_dir, exist_ok=True)

#%% load data
with open("misc/dataset/filtered_data.pkl", "rb") as f:
    ecg_signals, labels = pkl.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(['Other', 'Myocardial infarction'])


def normalize_signal(signal):
    return (signal - np.mean(signal, axis=1, keepdims=True)) / (np.std(signal, axis=1, keepdims=True) + 1e-6)


def create_subject_graph(signal: np.ndarray, label_str: str) -> Data:
    y = torch.tensor([label_encoder.transform([label_str])[0]], dtype=torch.long)
    # Dummy edge index: fully connected
    edge_index = torch.combinations(torch.arange(15), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=torch.tensor(signal, dtype=torch.float), edge_index=edge_index, y=y)

# graph_list = [create_subject_graph(sig, lbl) for sig, lbl in zip(ecg_signals, labels)]
graph_list = [create_subject_graph(normalize_signal(sig), lbl) for sig, lbl in zip(ecg_signals, labels)]

train_graphs, val_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)  # Batch size = 1 because lengths vary
val_loader = DataLoader(val_graphs, batch_size=1)

#%% Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_type == 'CGNN':
    model = ECGCGNN(cnn_output_dim=64, gnn_hidden_dim=64, num_classes=2).to(device)
elif model_type == 'RGNN':
    model = ECGRGNN(rnn_output_dim=64, gnn_hidden_dim=64, num_classes=2).to(device)
else:
    raise ValueError("Invalid model type. Choose 'CGNN' or 'RGNN'.")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

#%% Training and Evaluation
def train(model, loader):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for i, batch in enumerate(loader):
        if i % 10 == 0:
            print(f"Training batch {i}/{len(loader)}")
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()

            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


#%% Training script
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

best_val_acc = 0.0
best_model_path = f'{model_dir}/best_model.pth'

num_epochs = 50

for epoch in range(0, num_epochs):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        best_flag = "✅ BEST"
    else:
        best_flag = ""

    # Print nicely
    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} {best_flag}"
    )

# Save metrics
with open(f'{model_dir}/metrics.pkl', "wb") as f:
    pkl.dump({
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history
    }, f)
