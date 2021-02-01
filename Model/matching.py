import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
import torch.nn.functional as F

CLS = False



class FT_Match(nn.Module):
    def __init__(self, args):
        super(FT_Match, self).__init__()

        self.BertM = BertModel.from_pretrained('bert-base-chinese')
        self.BertM.training = True
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, bidirectional=True, batch_first=True)

        if CLS:
            list_MLP = [
                nn.Linear(768 * 2, 768), nn.LeakyReLU(),
                nn.Linear(768, 128), nn.LeakyReLU(), nn.Linear(128, 1)]
        else:
            list_MLP = [
                nn.Linear(768 * 4, 2 * 768), nn.LeakyReLU(),
                nn.Linear(768 * 2, 768), nn.LeakyReLU(),
                nn.Linear(768, 128), nn.LeakyReLU(), nn.Linear(128, 1)]

        self.MLP = nn.Sequential(*list_MLP)

    def forward(self, x):
        if CLS:
            sen_emb = self.BertM(x[0], attention_mask=x[1])[1]
            con_emb = self.BertM(x[2], attention_mask=x[3])[1]
            h_concept = con_emb
            h_sentence = sen_emb
        else:
            sen_emb = self.BertM(x[0], attention_mask=x[1])[0]
            con_emb = self.BertM(x[2], attention_mask=x[3])[0]
            h_concept = self.lstm(con_emb)[0][:, -1, :]
            h_sentence = self.lstm(sen_emb)[0][:, -1, :]

        h_cat = torch.cat((abs(h_sentence - h_concept), h_sentence.mul(h_concept)), -1)

        output = self.MLP(h_cat)
        return output
