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
        context = self.data['content'][index]
        encoder_pair = self.tokenizer(context, padding='max_length', truncation=True,
                                      max_length=self.maxlen, return_tensors='pt')
        token_ids = encoder_pair['input_ids'].squeeze(0)
        atten_masks = encoder_pair['attention_mask'].squeeze(0)
        token_type_ids = encoder_pair['token_type_ids'].squeeze(0)

        if self.with_labels:
            label = int(self.data['class_label'][index])
            return token_ids, atten_masks, token_type_ids, label
        else:
            return token_ids, atten_masks, token_type_ids
