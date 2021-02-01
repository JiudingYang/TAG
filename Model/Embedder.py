import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *
from configs import *


class Embedder(nn.Module):
    """
    Embedding different features according to configuration
    and concatenate all the embeddings.
    """

    def __init__(self, emb_dicts, emb_mats, dropout=0.1, _configs=None):
        """
        Initialize two dicts: self.embs and self.conv2ds.
        self.embs contains different Embedding layers for different tags.
        self.conv2ds contains different Conv2d layers for different tags
        that requires convolution, e.g., character or BPE embedding.
        :param config: arguments
        :param emb_mats: dict of embedding matrices for different tags
        :param emb_dicts: dict of word2id dicts for different tags.
        :param dropout: dropout rate for dropout layers after embedding and after convolution
        """
        super().__init__()
        if _configs is None:
            self.config = EmbConfig
        else:
            self.config = _configs.EmbConfig
        self.embs = torch.nn.ModuleDict()
        # self.conv2ds = torch.nn.ModuleDict()
        # construct all keys, so we reuse one embedder
        # and can train on different tasks
        for tag in self.config.emb_tags:
            if tag =='word':
                continue
            if self.config.emb_config[tag]["need_emb"]:
                self.embs.update(
                    {tag:
                        nn.Embedding.from_pretrained(
                            torch.tensor(emb_mats[tag]).float(),
                            freeze=(not self.config.emb_config[tag]["need_train"]))})

        self.dropout = dropout

    def forward(self, batch, emb_tags, device):
        """
        Given a batch of data, the field and tags we want to emb,
        return the concatenated embedding representation.
        :param batch: a batch of data. It is a dict of tensors.
            Each tensor is tag ids or tag values.
            Input shape - [batch_size, seq_length]
        :param emb_tags: a list of tags to indicate which tags we will embed
        :return: concatenated embedding representation
            Output shape - [batch_size, emb_dim, seq_length]
        """
        emb = torch.FloatTensor().to(device)
        # print("batch is:::: ", batch)
        # use emb_tags to control which tags are actually in use
        for tag in emb_tags:
            if tag == 'word':
                continue
            if self.config.emb_config[tag]["need_emb"]:
                tag_emb = self.embs[tag](batch[tag])
            else:
                tag_emb = batch[tag].unsqueeze(2)
            tag_emb = F.dropout(
                tag_emb, p=self.dropout, training=self.training)

            # print("tag_emb shape ", tag_emb.shape)
            emb = torch.cat([emb, tag_emb], dim=-1)
        return emb
