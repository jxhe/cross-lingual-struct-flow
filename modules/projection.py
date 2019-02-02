from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class ReLUNet(nn.Module):
    def __init__(self, hidden_layers, hidden_units, in_features, out_features):
        super(ReLUNet, self).__init__()

        self.hidden_layers = hidden_layers
        self.in_layer = nn.Linear(in_features, hidden_units, bias=True)
        self.out_layer = nn.Linear(hidden_units, out_features, bias=True)
        for i in range(hidden_layers):
            name = 'cell{}'.format(i)
            cell = nn.Linear(hidden_units, hidden_units, bias=True)
            setattr(self, name, cell)

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        self.in_layer.weight.data.zero_()
        self.in_layer.bias.data.zero_()
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).weight.data.zero_()
            getattr(self, name).bias.data.zero_()

    def forward(self, input):
        """
        input: (batch_size, seq_length, in_features)
        output: (batch_size, seq_length, out_features)

        """
        h = self.in_layer(input)
        h = F.relu(h)
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            h = getattr(self, name)(h)
            h = F.relu(h)
        return self.out_layer(h)


class NICETrans(nn.Module):
    def __init__(self,
                 couple_layers,
                 cell_layers,
                 hidden_units,
                 features,
                 device):
        super(NICETrans, self).__init__()

        self.device = device
        self.couple_layers = couple_layers

        for i in range(couple_layers):
            name = 'cell{}'.format(i)
            cell = ReLUNet(cell_layers, hidden_units, features//2, features//2)
            setattr(self, name, cell)

    def reset_parameters(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).init_identity()


    def forward(self, input, masks=None):
        """
        input: (seq_length, batch_size, features)
        h: (seq_length, batch_size, features)

        """

        # For NICE it is a constant
        jacobian_loss = torch.zeros(1, device=self.device,
                                    requires_grad=False)

        ep_size = input.size()
        features = ep_size[-1]
        # h = odd_input
        h = input
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            h1, h2 = torch.split(h, features//2, dim=-1)
            if i%2 == 0:
                h = torch.cat((h1, h2 + getattr(self, name)(h1)), dim=-1)
            else:
                h = torch.cat((h1 + getattr(self, name)(h2), h2), dim=-1)
        return h, jacobian_loss


class LSTMNICE(nn.Module):
    def __init__(self, n_lstm_layer, n_couple_layer,
                 n_relu_layer, n_hid_lstm, n_hid_nice,
                 n_emb, device):

        super(LSTMNICE, self).__init__()
        self.n_lstm_layer = n_lstm_layer
        self.n_couple_layer = n_couple_layer
        self.n_hid_lstm = n_hid_lstm
        self.n_hid_nice = n_hid_nice
        self.n_emb = n_emb
        self.device = device

        couple_layers = []
        for _ in range(n_couple_layer):
            couple_layers.append(ReLUNet(n_relu_layer, n_hid_nice, n_emb // 2 + n_hid_lstm, n_emb // 2))

        self.couple_layers = nn.ModuleList(couple_layers)

        self.lstm = nn.LSTM(input_size=n_emb,
                            hidden_size=n_hid_lstm,
                            num_layers=n_lstm_layer)

        # embedding of start symbol
        self.start_emb = nn.Parameter(torch.Tensor(n_emb).uniform_(-0.01, 0.01))

    def forward(self, input, masks=None):
        """
        input: (seq_length, batch_size, features)
        masks: (seq_length, batch_size)
        output: (seq_length, batch_size, features)
        """
        jacobian_loss = torch.zeros(1, requires_grad=False,
                                    device=self.device)

        batch_size = input.size(1)
        pad_start = self.start_emb.view(1, 1, self.n_emb).expand(1, batch_size, self.n_emb)
        pad_input = torch.cat((pad_start, input), dim=0)

        if masks is not None:
            # consider the padded start
            sents_len = (masks.sum(dim=0).data.long() + 1).tolist()
            packed_input = pack_padded_sequence(pad_input, sents_len)
        else:
            packed_input = pad_input

        # (seq_len + 1, batch_size, hidden_size)
        output, _ = self.lstm(packed_input)

        if masks is not None:
            output, _ = pad_packed_sequence(output)

        # (seq_len, batch_size, hidden_size)
        output = output[:-1, :, :]
        h = input
        for i in range(self.n_couple_layer):
            h1, h2 = torch.chunk(h, 2, dim=-1)
            if i%2 == 0:
                h = torch.cat((h1, h2 + self.couple_layers[i](torch.cat((h1, output), dim=-1))), dim=-1)
            else:
                h = torch.cat((h1 + self.couple_layers[i](torch.cat((h2, output), dim=-1)), h2), dim=-1)


        return h + output, jacobian_loss
