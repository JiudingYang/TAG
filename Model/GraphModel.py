import torch
import torch.nn as nn
import torch.nn.functional as F
import Model.config
from torch_geometric.nn import RGCNConv
from .Embedder import Embedder
import configs as ori_configs


class TAG(nn.Module):

    def __init__(self, emb_dict, emb_mats, dropout=0.1, reg=0, configs=ori_configs):
        super(TAG, self).__init__()
        self.in_channels = configs.EmbConfig.emb_len_total
        self.out_channels = self.in_channels
        self.num_relations = configs.ModelConfig.num_relations
        self.num_bases = configs.ModelConfig.num_bases
        self.num_layers = configs.ModelConfig.num_layers
        self.dropout = dropout
        self.rgcn_layers = nn.ModuleList()
        self.reg = reg
        for i in range(self.num_layers):
            self.rgcn_layers.append(
                RGCNConv(self.in_channels, self.out_channels, self.num_relations, num_bases=self.num_bases))

        self.embedder = Embedder(emb_dict, emb_mats)
        list_MLP = [nn.Linear(240 * 2, 128), nn.LeakyReLU(), nn.Linear(128, 1)]
        self.MLP = nn.Sequential(*list_MLP)

    def forward(self, x, emb_idx_dict, edge_index, edge_type, configs=ori_configs, edge_norm=None):
        """
        :param configs:
        :param x: list of tensor vectors, glove embedding of nodes
        :param emb_idx_dict: dict with tags as keys, each value of a key is the index sequence of all nodes.
        :param edge_index: rgcn index of edge, list of list
        :param edge_type: rgcn type idx of edge, list
        :param edge_norm: rgcn norm
        :return:
        """
        x = torch.cat([x.float(), self.embedder(emb_idx_dict, configs.EmbConfig.emb_tags, configs.DEVICE)], dim=-1)
        concept_index = -1
        sentence_index = -1
        for index, p in enumerate(emb_idx_dict['pos']):
            if p == 0:
                concept_index = index
            if p == 1:
                sentence_index = index

        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                x = F.dropout(F.leaky_relu(self.rgcn_layers[i](x, edge_index, edge_type)), p=self.dropout,
                              training=self.training)
            else:
                x = self.rgcn_layers[i](x, edge_index, edge_type)

        h_sentence = x[sentence_index]
        h_concept = x[concept_index]
        h_cat = torch.cat((abs(h_sentence - h_concept), h_sentence.mul(h_concept)), -1)
        output = self.MLP(h_cat)
        return output
