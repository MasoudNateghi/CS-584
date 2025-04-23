import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class LeadRNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(LeadRNN, self).__init__()
        self.gru = nn.GRU(
            input_size=1,         # Each timestep is a scalar (ECG value)
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False   # Set to True if you want bidirectional GRU
        )

    def forward(self, lead_signals):
        # lead_signals: [N, signal_length] — where N = total number of leads in the batch
        x = lead_signals.unsqueeze(-1)  # Shape becomes [N, signal_length, 1]
        _, h_n = self.gru(x)            # h_n: [1, N, hidden_dim] (or [2, N, hidden_dim] if bidirectional)
        return h_n.squeeze(0)           # Output: [N, hidden_dim] — per-lead encoded feature


class ECGRGNN(nn.Module):
    def __init__(self, rnn_output_dim, gnn_hidden_dim, num_classes):
        super(ECGRGNN, self).__init__()
        self.lead_encoder = LeadRNN(hidden_dim=rnn_output_dim)
        self.gnn1 = GCNConv(rnn_output_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.fc = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, data):
        x_raw, edge_index, batch = data.x, data.edge_index, data.batch
        x_encoded = self.lead_encoder(x_raw)  # [15, rnn_output_dim]
        x = torch.relu(self.gnn1(x_encoded, edge_index))
        x = torch.relu(self.gnn2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)