import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


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


class ECGattCGNN(nn.Module):
    def __init__(self, cnn_output_dim, gnn_hidden_dim, num_classes, num_heads=4):
        super(ECGattCGNN, self).__init__()
        self.lead_encoder = LeadCNN(output_dim=cnn_output_dim)

        # Attention layer: output dim = hidden_dim * num_heads
        self.gnn1 = GATConv(cnn_output_dim, gnn_hidden_dim, heads=num_heads, concat=True)
        self.gnn2 = GATConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, heads=1, concat=False)

        self.fc = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, data):
        x_raw, edge_index, batch = data.x, data.edge_index, data.batch

        x_encoded = self.lead_encoder(x_raw)  # [15 * batch_size, cnn_output_dim]

        x = torch.relu(self.gnn1(x_encoded, edge_index))  # Apply attention
        x = torch.relu(self.gnn2(x, edge_index))
        x = global_mean_pool(x, batch)  # Aggregate node features per graph

        return self.fc(x)
