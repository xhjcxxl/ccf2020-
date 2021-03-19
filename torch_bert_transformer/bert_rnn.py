# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel


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
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        encoder_out, text_cls = self.bert(context, token_type_ids=token_type_ids, attention_mask=mask)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        logits = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return logits
