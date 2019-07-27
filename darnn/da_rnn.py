"""
da-rnn
Based on the Chandler Zuo git
https://arxiv.org/pdf/1704.02971.pdf
http://chandlerzuo.github.io/blog/2017/11/darnn
"""

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderDARNN(nn.Module):
    def __init__(self, input_size, hidden_size, T):
        super(EncoderDARNN, self).__init__()
        # input_size = # of underlying factors (81)
        # T: number of time steps
        # hidden_size: dimension of hidden states
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        # lstm layer
        self.lstm_layer = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=1)

        # activation
        self.attn_linear = nn.Sequential(
            nn.Linear(in_features=2*hidden_size + T - 1,
                      out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, X):

        # X: batch_size * T-1 * input_size
        input_weighted = Variable(X.data.new(X.size(0), self.T-1, self.input_size).zero_())
        input_encoded = Variable(X.data.new(X.size(0), self.T-1, self.hidden_size).zero_())
        # hidden, cell: initial states with dimension hidden_size
        hidden = self.init_hidden(X) # 1 * batch_size * hidden_size
        cell = self.init_hidden(X)

        # iterate through time steps
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)

            # Eqn. 9: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))  # (batch_size * input_size) * 1
            attn_weights = F.softmax(x.view(-1, self.input_size))

            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, X[:, t, :])  # batch_size * input_size

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension


class DecoderDARNN(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(DecoderDARNN, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.T = T

        # attention layer
        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, 1)
        )

        # lstm
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_hidden_size)

        # fully connected layer
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)

        # final fully connected layer
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        """
        input_encoded: batch_size * T-1 * encoder_hidden_size
        y_history: batch_size * (T-1)
        """

        # initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)

        # iterate through time steps
        for t in range(self.T - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T-1 * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim=2)
            x = self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size))
            # softmax layer
            x = F.softmax(x.view(-1, self.T - 1))  # batch_size * T - 1, row sum up to 1

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # batch_size * encoder_hidden_size

            if t < self.T - 1:
                # Eqn. 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim=1))  # batch_size * 1

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim=1))

        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())