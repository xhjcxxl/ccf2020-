# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # 1024*1024
        self.weight.data.normal_(mean=0.0, std=0.05)  # 初始化

        self.bias = nn.Parameter(torch.Tensor(hidden_size))  # 1024*1

        b = np.zeros(hidden_size, dtype=np.float32)  # 初始设定
        self.bias.data.copy_(torch.from_numpy(b))  # 初始化

        self.query = nn.Parameter(torch.Tensor(hidden_size))  # 应该看作 1024*1
        self.query.data.normal_(mean=0.0, std=0.05)  # 初始化

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: batch x length x hidden_size (2 * hidden_size of lstm) 这里正好就是BiLSTM模型的输出结果
        # batch_masks:  batch x length 这个就是数据的输入mask

        # key是encoder的各个隐藏状态，就是LSTM的隐藏状态结果
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # batch x length x hidden 8*512*1024

        # compute attention (Q,K)结果就是score得分
        outputs = torch.matmul(key, self.query)  # batch x length 8*512
        # 把output中的对应位置的数字mask为1
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))  # 8*512

        # 进行softmax
        attn_scores = F.softmax(masked_outputs, dim=1)  # batch x length 进行softmax 8*512

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)  # 8*512
        # sum weighted sources (8*1*512 X 8*512*1024=8*1*1024)，然后压缩一个维度变成了8*1024
        # 再用encoder的隐藏状态乘以sotmax之后的得分
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


class bert_RNN(nn.Module):
    def __init__(self, config):
        super(bert_RNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.attention = Attention(config.rnn_hidden * 2)  # 创建attention函数 输入的就是 1024（两个rnn_hidden）
        self.linear = nn.Sequential(
            nn.Linear(config.rnn_hidden * 2, config.rnn_hidden),  # 这里是rnn_hidden * 2是因为使用了BiLSTM
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rnn_hidden, config.num_classes)
        )

    def forward(self, x):
        context = x[0]  # 输入句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # 因为我选择了参数output_all_encoded_layers=True，12层Transformer的结果全返回了，存在第一个列表中，
        # 每个encoder_output的大小为[batch_size, sequence_length, hidden_size]
        context = torch.squeeze(context, dim=1)
        mask = torch.squeeze(mask, dim=1)
        # output_all_encoded_layers 取 bert 最后一层
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        lstm_hidden, _ = self.lstm(encoder_out)  # 8*512*1024

        lstm_hiddens = lstm_hidden * mask.unsqueeze(2)  # 8*512*1024
        # lstm_hiddens = self.dropout(lstm_hiddens)  # 8*512*1024
        out, atten_scores = self.attention(lstm_hiddens, mask)  # batch x rnn_hidden 返回结果是8*1024
        out = self.dropout(out)  # 8*1024

        logits = self.linear(out)  # 8*1024 句子最后时刻的 hidden state
        return logits
