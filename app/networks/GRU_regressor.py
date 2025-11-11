import torch
import torch.nn as nn

class GRURegressor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 out_dim: int,
                 dropout: float = 0.0,
                 batch_first: bool = True):
        super(GRURegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        # GRU layer
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Fully-connected (linear) layer for regression output
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, seq_len, in_dim) if batch_first=True
        returns: tensor of shape (batch_size, out_dim)
        """
        # Pass through GRU
        # out: (batch_size, seq_len, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        out, h_n = self.gru(x)
        # Take the last hidden state of the last layer
        last_hidden = h_n[-1]  # shape: (batch_size, hidden_dim)
        # Regression output
        output = self.fc(last_hidden)  # shape: (batch_size, out_dim)
        return output




if __name__ == '__main__':
    in_dim = 10       # number of features per time step
    hidden_dim = 64   # hidden size of GRU
    num_layers = 2    # number of stacked GRU layers
    out_dim = 3       # for a single continuous target value
    model = GRURegressor(in_dim, hidden_dim, num_layers, out_dim, dropout=0.3, batch_first=True)

    batch_size = 32
    seq_len = 20
    x = torch.randn(batch_size, seq_len, in_dim)
    y_pred = model(x)
    print(y_pred.shape)  # → torch.Size([32, 3])
