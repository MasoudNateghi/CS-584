import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the filtered records and labels
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

graph_list = [create_subject_graph(sig, lbl) for sig, lbl in zip(ecg_signals, labels)]

train_graphs, val_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)  # Batch size = 1 because lengths vary
val_loader = DataLoader(val_graphs, batch_size=1)


#%% Model Definition
class LeadCNN(nn.Module):
    def __init__(self, output_dim):
        super(LeadCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.linear = nn.Linear(32, output_dim)

    def forward(self, lead_signals):
        # lead_signals: [15, signal_length]
        x = lead_signals.unsqueeze(1)  # [15, 1, signal_length]
        x = self.encoder(x).squeeze(-1)  # [15, 32]
        x = self.linear(x)  # [15, output_dim]
        return x

class HybridECGGNN(nn.Module):
    def __init__(self, cnn_output_dim, gnn_hidden_dim, num_classes):
        super(HybridECGGNN, self).__init__()
        self.lead_encoder = LeadCNN(output_dim=cnn_output_dim)
        self.gnn1 = GCNConv(cnn_output_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.fc = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, data):
        x_raw, edge_index, batch = data.x, data.edge_index, data.batch
        x_encoded = self.lead_encoder(x_raw)  # [15, cnn_output_dim]
        x = torch.relu(self.gnn1(x_encoded, edge_index))
        x = torch.relu(self.gnn2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

#%% Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridECGGNN(cnn_output_dim=64, gnn_hidden_dim=64, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

#%% Training and Evaluation
def train(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

#%% Training script
best_val_acc = 0.0
best_model_path = "models/CNNGCN/best_model.pth"

num_epochs = 100
for epoch in range(0, num_epochs):
    train_loss = train(model, train_loader)
    val_acc = evaluate(model, val_loader)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} [BEST SAVED]")
    else:
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
