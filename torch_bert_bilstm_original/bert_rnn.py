# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
import json


class bert_RNN(nn.Module):

    def __init__(self, config):
        super(bert_RNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)  # 这里是rnn_hidden * 2是因为使用了BiLSTM

    def forward(self, x):
        context = x[0]  # 输入句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # 因为我选择了参数output_all_encoded_layers=True，12层Transformer的结果全返回了，存在第一个列表中，
        # 每个encoder_output的大小为[batch_size, sequence_length, hidden_size]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
