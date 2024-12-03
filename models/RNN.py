import torch
import torch.nn as nn
from lib.utils import *
from lib.modules import *

@dataclass(kw_only=True)
class RNNConfig(Config):
    features: int
    hidden_size: int
    layers: int
    flavor: str = "rnn"
    dropout: float = 0.2
    y_len: int = 1


class RNN(Module):
    def __init__(self, c: RNNConfig):
        """
        Parameters
        ----------
        features: int
            This is the same as number of features in traditional lingo.
            For univariate time series, this would be 1 and greater than 1
            for multivariate time series.
        hidden_size: int
            Number of hidden units in the RNN model
        layers: int
            Number of layers in the RNN model
        flavor: str
            Takes 'rnn', 'lstm', or 'gru' values.
        y_len: int
            Number of most recent predictions to output (For experimentation, todo: slices)
        """

        super().__init__()
        self.save_config(c)
        dropout = c.dropout if c.layers > 1 else 0

        self.layernorm = nn.LayerNorm(c.features)
        # batch first changes in and out dimensions to (batch_size, num_steps, features/hidden_size)
        if c.flavor == "lstm":
            self.rnn = nn.LSTM(
                c.features, c.hidden_size, c.layers, dropout=dropout, batch_first=True
            )
        elif c.flavor == "gru":
            self.rnn = nn.GRU(
                c.features, c.hidden_size, c.layers, dropout=dropout, batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                c.features, c.hidden_size, c.layers, dropout=dropout, batch_first=True
            )

        self.fc = nn.Sequential(
            nn.Linear(c.hidden_size, 1),
        )

    def forward(self, x, state=None):
        x = self.layernorm(x)
        if state is None:
            state = torch.zeros(
                self.layers, x.size(0), self.hidden_size, device=x.device
            )
            if self.flavor == "lstm":
                state = (state, torch.zeros_like(state, device=x.device))  # cell_state

        out, _ = self.rnn(
            x, state
        )  # updates and appends state for each element of batch
        out = self.fc(out[:, -self.y_len, :])
        return out

    def predict_n(self, input_seq, n):
        pass
