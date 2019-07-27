"""
lstm
based on ST-RNN lstm decoder
https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-38.pdf
"""

import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, T, num_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input_size # num_features 81
        self.hidden_size = hidden_size # 100
        self.T = T
        self.num_layers = num_layers

        # lstm in: (N, T, W) out: (N, T, H)
        self.lstm_layer = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=0.2,
                                  batch_first=True)
        # dense layer in: (N, T*H) out: (N, T*H)
        self.dense_layer = nn.Sequential(
            nn.Linear(T*(hidden_size + 1), T*(hidden_size + 1)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # final layer in: (N, T*H) out: (N, 1)
        self.final_layer = nn.Linear(T*(hidden_size + 1), 1)

    def forward(self, x, y_history):
        """
        x: (N, T, W)
        y_history: (N, T, 1)
        """

        # lstm in: (N, T, W) out: (N, T, H)
        out, lstm_out = self.lstm_layer(x)
        # clip gradients
        out.register_hook(lambda x: x.clamp(min=-100, max=100))
        # combine with y_history
        out = torch.cat((out, y_history), dim=2)
        # flatten in: (N, T, H) out: (N, T*(H+1))
        out = out.contiguous().view(-1, out.size(1) * out.size(2))
        # dense layer in: (N, T*(H+1)) out: (N, T*(H+1))
        out = self.dense_layer(out)
        # final layer
        out = self.final_layer(out)

        return out

    def init_hidden(self, x):
        # hidden shape (num_layers * num_directions, batch, hidden_size)
        return Variable(x.data.new(self.num_layers, x.size(1), self.hidden_size))