# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import shuffle

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 加载数据
def load_data(filename, config, with_label=True):
    data = pd.read_csv(filename, encoding='utf-8')
    print(len(data))
    if with_label:
        data = data.replace({'class_label': config.label2idx})
    return data


def train_test_split(data_df, test_size=0.2, shuffle_flag=True, random_state=None):
    if shuffle_flag:
        data_df = shuffle(data_df, random_state=random_state)  # random_state保证每次划分结果一样

    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    valid = data_df[: int(len(data_df) * test_size)].reset_index(drop=True)

    return train, valid


class Mydataset(Dataset):
    def __init__(self, config, data, with_labels=True):
        self.config = config
        self.data = data
        self.tokenizer = config.tokenizer
        self.maxlen = config.pad_size
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # context = str(self.data.loc[index, 'context'])
        context = self.data['content'][index]
        token = self.tokenizer.tokenize(context)
        token = ['[CLS]'] + token
        seq_len = len(token)
        mask = []
        token_ids = self.tokenizer.convert_tokens_to_ids(token)

        # padding
        if seq_len < self.maxlen:
            token_ids += ([0] * (self.maxlen - seq_len))
            mask_ids = [1] * seq_len + [0] * (self.maxlen - seq_len)
            segment_ids = [0] * seq_len + [0] * (self.maxlen - seq_len)
        else:
            token_ids = token_ids[:self.maxlen]
            mask_ids = [1] * self.maxlen
            segment_ids = [0] * self.maxlen

        token_ids = torch.LongTensor([token_ids]).to(self.config.device)
        segment_ids = torch.LongTensor([segment_ids]).to(self.config.device)
        mask_ids = torch.LongTensor([mask_ids]).to(self.config.device)

        if self.with_labels:
            label = int(self.data['class_label'][index])
            label = torch.LongTensor([label]).to(self.config.device)
            return token_ids, segment_ids, mask_ids, label
        else:
            return token_ids, segment_ids, mask_ids
