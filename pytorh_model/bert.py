# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertModel, BertTokenizer
import json


class CLSModel(nn.Module):

    def __init__(self, config):
        super(CLSModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(input_size=embedding_dim * 4, hidden_size=hidden_dim,
                          batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.linear = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 2, out_features=512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=config.num_classes)
        )
        self.dropout = nn.Dropout(p=dropout)

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        input_ids = x[0]
        token_type_id = x[1]
        attn_masks = x[2]
        embedding = self.bert(input_ids, token_type_id=token_type_id, attn_masks=attn_masks)
        hidden_stats = torch.cat(tuple([embedding.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1)
        first_hidden_states = hidden_stats[:, 0, :]
        first_hidden_states = first_hidden_states.unsqueeze(dim=1)
        _, hidden = self.rnn(first_hidden_states)
        hidden = self.dropout(torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1) if self.rnn.bidirectional else hidden[-1, :, :])
        logits = self.linear(hidden)
        return logits


class bert_model(nn.Module):

    def __init__(self, config):
        super(bert_model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
