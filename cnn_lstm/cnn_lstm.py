"""
image-captioning model
encoder: cnn
decoder: lstm
"""

import torch
from torch import nn
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, input_dim=1, channel_size=64, batch_size=10,
                 T=100, feature_size=81):
        super(EncoderCNN, self).__init__()
        self.input_dim = input_dim
        self.channel_size = channel_size
        self.batch_size = batch_size
        self.T = T
        self.feature_size = feature_size
        self.modelType = "encoder"
        # (N, C, H, W) = (num_batch, features, history, stocks)

        # added a linear layer to shrink the num stocks lower due to memory
        self.small_feature_size = 10
        self.first_linear = nn.Linear(feature_size, self.small_feature_size)

        # Conv2d - out:(N, 64, 100, 81), kernel:(3,5), stride:1
        self.first_cnn_layer = nn.Sequential(
            nn.Conv2d(input_dim, channel_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Conv2d - out:(N, 64, 100, 81), kernel:(3,5), stride:1
        self.second_cnn_layer = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # dense layer - in:(N, 100*64*81), out:(N, 100*81)
        self.first_dense_layer = nn.Sequential(
            nn.Linear(T * self.small_feature_size * channel_size, T * self.feature_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, xt):
        # conv2d input:(N, 1, H, W) expects (N, C, H, W)
        N = xt.size(0)
        # linear: in(N, 1, H, W) out: (N, 1, H, 10)
        out = self.first_linear(xt)
        # cnn: in(N, 1, H, 10) out: (N, C, H, 10)
        out = self.first_cnn_layer(out)
        # cnn: in(N, C, H, 10), out: (N, C, H, 10)
        out = self.second_cnn_layer(out)
        # reshape for linear layer
        out = out.view(N, self.T*self.small_feature_size*self.channel_size)
        # first dense layer in: (N, C*H*W) out: (N, H*W)
        out = self.first_dense_layer(out)
        # reshape output for (N, T, W)
        out = out.reshape(out.size(0), self.T, self.feature_size)

        return out

class DecoderLSTM(nn.Module):
    def __init__(self, feature_size, decoder_hidden_size, T=100, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.feature_size = feature_size
        self.decoder_hidden_size = decoder_hidden_size
        self.T = T
        self.num_layers = num_layers
        self.modelType = "decoder"

        # lstm - in: (N, T, W) out: (N, T, H)
        self.lstm_layer = nn.LSTM(feature_size, decoder_hidden_size,
                                  num_layers=num_layers,
                                  dropout=0.2,
                                  batch_first=True)

        # dense layer - in: (N, T*H), out: (N, T*H)
        self.dense_layer = nn.Sequential(
            nn.Linear(T*(decoder_hidden_size + 1), T*(decoder_hidden_size + 1)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # final layer - in: (N, T*H) out: (N, 1)
        self.final_layer = nn.Linear(T*(decoder_hidden_size + 1), 1)

    def forward(self, features, y_history):

        # lstm in: (N, T, W) out: (N, T, H)
        out, lstm_out = self.lstm_layer(features)
        # clipping to eliminate nan's from lstm
        out.register_hook(lambda x: x.clamp(min=-100, max=100))
        # combine with y_history
        out = torch.cat((out, y_history), dim=2)
        # flatten in: (N, T, H) out: (N, T*(H+1))
        out = out.contiguous().view(-1, out.size(1) * out.size(2))
        # final layer in: (N, T*(H+1)), out: (N, 1)
        out = self.final_layer(out)

        return out

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero())