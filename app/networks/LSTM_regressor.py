import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self,
                 in_dim:int,
                 hidden_dim:int,
                 out_dim:int,
                 num_layers:int=1,
                 batch_first:bool=True,
                 dropout:float=0.0,
                 bidirectional:bool=False):
        """
        LSTM-based regression model.

        **Arguments**:
        - `in_dim` : Number of input features per time step.
        - `hidden_dim` : Number of hidden units in the LSTM.
        - `out_dim` : Dimensionality of the regression output vector.
        - `num_layers` : Number of stacked LSTM layers.
        - `batch_first` : Whether input shape is (batch, seq_len, in_dim) or (seq_len, batch, in_dim).
        - `dropout` : Dropout probability (only applies between LSTM layers if num_layers > 1).
        - `bidirectional` : Whether to use a bidirectional LSTM.
        """
        super(LSTMRegressor, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional
                           )
        # After LSTM we take the last time-step (or hidden state) and map to output
        self.fc = nn.Linear(hidden_dim * self.num_directions, out_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        **Arguments**:
        - `x` : Input tensor of shape (`batch`, `seq_len`, `in_dim`) if batch_first=True,
            else (`seq_len`, `batch`, `in_dim`).

        **Returns**:
        - `x` : Output tensor of shape (`batch`, `out_dim`)
        """
        # Initialize hidden state and cell state (optional; PyTorch will default to zeros)
        batch_size = x.size(0) if self.batch_first else x.size(1)
        h0 = torch.zeros(self.num_layers * self.num_directions,
                         batch_size,
                         self.hidden_dim,
                         device=x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions,
                         batch_size,
                         self.hidden_dim,
                         device=x.device)
        # LSTM forward
        out_seq, (hn, cn) = self.lstm(x, (h0, c0))
        # out_seq shape: (batch, seq_len, hidden_dim * num_directions) if batch_first
        # We choose to use the last time‐step's output:
        if self.batch_first:
            last_out = out_seq[:, -1, :]
        else:
            last_out = out_seq[-1, :, :]
        # Fully connected regression output
        y = self.fc(last_out)
        return y
    




if __name__ == '__main__':

    import utils.utils as utils
    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "yellow"

    params:dict = utils.load_json("/data/params.json")

    # Example parameters
    in_dim = 10       # e.g., 10 input features per time-step
    hidden_dim = 64   # number of hidden units in LSTM
    out_dim = 5       # we want to predict a 5-dimensional vector
    num_layers = 2
    batch_first = True

    model = LSTMRegressor(in_dim, hidden_dim, out_dim, num_layers, batch_first)
    print(model)

    # Example dummy input: batch of 32 sequences, each length 20
    x = torch.randn(32, 20, in_dim)
    y_pred = model(x)  # shape should be (32, out_dim)
    print(y_pred.shape)

