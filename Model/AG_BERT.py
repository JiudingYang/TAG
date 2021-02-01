from transformers import BertModel
import torch.nn as nn
import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F


class AG_Match(nn.Module):
    def __init__(self, layers, dropout=0.1):
        super(AG_Match, self).__init__()
        self.in_channels = 200
        self.out_channels = 200
        self.num_relations = 2
        self.num_bases = 2
        self.num_layers = layers
        self.dropout = dropout
        self.rgcn_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.rgcn_layers.append(
                RGCNConv(self.in_channels, self.out_channels, self.num_relations, num_bases=self.num_bases))
        # self.lstm = nn.LSTM(input_size=200, hidden_size=200, bidirectional=True, batch_first=True)
        list_MLP = [
            # nn.Linear(200 * 4, 200*2), nn.LeakyReLU(),
            nn.Linear(200 * 2, 200), nn.LeakyReLU(),
            nn.Linear(200, 1)]
        self.MLP = nn.Sequential(*list_MLP)

    def forward(self, x, y, edge_index, edge_type, edge_norm=None):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                x = F.dropout(F.leaky_relu(self.rgcn_layers[i](x, edge_index, edge_type)), self.dropout,
                              training=self.training)
            else:
                x = self.rgcn_layers[i](x, edge_index, edge_type)

        h_concept = x[0]
        h_sentence = y

        h_cat = torch.cat((abs(h_sentence - h_concept), h_sentence.mul(h_concept)), -1)
        output = self.MLP(h_cat)

        return output
