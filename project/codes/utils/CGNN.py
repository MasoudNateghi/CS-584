import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


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

class ECGCGNN(nn.Module):
    def __init__(self, cnn_output_dim, gnn_hidden_dim, num_classes):
        super(ECGCGNN, self).__init__()
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