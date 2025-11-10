import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self,
                 in_dim:int,
                 hidden_dim:int,
                 num_layers:int,
                 out_dim:int,
                 dropout:float=0.0,
                 batch_first:bool=True
                ):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim) if batch_first=True
        out, (hn, cn) = self.lstm(x)
        if self.batch_first:
            last_hidden = out[:, -1, :]  # (batch_size, hidden_dim)
        else:
            last_hidden = out[-1, :, :]  # (batch_size, hidden_dim)
        logits = self.fc(last_hidden)   # (batch_size, out_dim)

        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs





if __name__ == '__main__':
    in_dim = 10
    hidden_dim = 64
    num_layers = 2
    out_dim = 5
    model = LSTMClassifier(in_dim, hidden_dim, num_layers, out_dim, dropout=0.3, batch_first=True)

    batch_size = 32
    seq_len = 20
    x = torch.randn(batch_size, seq_len, in_dim)
    probs = model(x)
    print(probs.shape)                  # torch.Size([32, 5])
    print(probs.sum(dim=1))             # each should be very close to 1
    print((probs >= 0).all().item())    # True (all non-negative)
    print(probs)
