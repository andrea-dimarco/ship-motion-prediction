import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUClassifier(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 out_dim: int,
                 dropout: float = 0.0,
                 batch_first: bool = True):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        # GRU layer
        self.gru = nn.GRU(
            input_size = in_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = batch_first,
            dropout = dropout if num_layers > 1 else 0.0
        )
        # Fully-connected layer for classification
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x : tensor of shape (batch_size, seq_len, in_dim) if batch_first=True
        """
        # Pass through GRU
        # out: tensor of shape (batch_size, seq_len, hidden_dim)
        # h_n: tensor of shape (num_layers, batch_size, hidden_dim)
        out, h_n = self.gru(x)
        # Use last hidden state of last layer:
        # If batch_first=True: h_n[-1] is (batch_size, hidden_dim)
        last_hidden = h_n[-1]  # shape: (batch_size, hidden_dim)
        # Pass through linear layer to get logits
        logits = self.fc(last_hidden)  # shape: (batch_size, out_dim)
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        return probs
