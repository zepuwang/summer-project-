import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    A simple version of Graph Attention Networks. The original version is from: https://arxiv.org/abs/1710.10903.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :]).view(N, -1, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :]).view(N, -1, 1)
        e = Wh1 + Wh2.transpose(1, 2)  # transpose for broadcasting
        return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        seq_length=50,
    ):
        super(TemporalBlock, self).__init__()

        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Calculate padding to retain the sequence length
        padding = (
            self.seq_length
            + (self.conv1.kernel_size[0] - 1) * self.conv1.dilation[0]
            - self.seq_length
        ) // 2
        out = F.pad(x, (padding, padding))
        out = self.relu(self.conv1(out))

        padding = (
            self.seq_length
            + (self.conv2.kernel_size[0] - 1) * self.conv2.dilation[0]
            - self.seq_length
        ) // 2
        out = F.pad(out, (padding, padding))
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


class TemporalConvNet(nn.Module):
    """
    The simple version of Temporal Convolutional Networks.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=1, seq_length=50):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size, dilation_size, seq_length
                )
            ]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Conv1d(num_channels[-1], 1, kernel_size)

    def forward(self, x):
        batch_size, num_nodes, seq_length, input_size = x.size()
        x = x.view(batch_size * num_nodes, seq_length, input_size)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = self.linear(x)
        return x.view(batch_size, num_nodes, seq_length, -1)
