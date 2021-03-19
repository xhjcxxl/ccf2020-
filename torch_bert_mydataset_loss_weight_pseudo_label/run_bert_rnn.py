# coding: UTF-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import torch
import numpy as np
import json
from train_eval import train, init_network
from importlib import import_module
import argparse
from pytorch_pretrained_bert import BertTokenizer
from bert_rnn import bert_RNN
from new_utils import Mydataset, get_time_dif, load_data, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/all_label_train_3class_1.csv'  # 训练集
        self.dev_path = dataset + '/all_label_valid_3class_1.csv'  # 训练集
        self.test_path = dataset + '/test_data.csv'  # 测试集
        self.id2label_path = dataset + '/id2label.json'  # label集合
        self.idx2label, self.label2idx = json.load(open(self.id2label_path, encoding='utf-8'))  # 类别名单
        self.save_path = dataset + '/saved_dict_torch/' + self.model_name + '0103_mydataset_loss_weight_pseudo.bin'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 100000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.idx2label)  # 类别数
        self.num_epochs = 5  # epoch数
        self.batch_size = 8  # mini-batch大小
        self.pad_size = 512  # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5  # 学习率
        self.bert_path = '../bert/torch_bert_chinese/'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.hidden_size = 768
        self.seed = 2021520
        self.rnn_hidden = 512
        self.dropout = 0.2
        self.num_layers = 2
        self.label2idx = {key: int(value) for key, value in self.label2idx.items()}
        self.idx2label = {int(key): value for key, value in self.idx2label.items()}


if __name__ == '__main__':
    dataset = '../dataset'  # 数据集
    config = Config(dataset)
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config.train_path = dataset + '/loss_weight/merge_all_label_unlabel_data_loss_weight.csv'
    data_df = load_data(config.train_path, config, with_label=True)
    train_df, valid_df = train_test_split(data_df, test_size=0.2, shuffle_flag=True, random_state=seed)

    # weights = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 0.4, 2, 1])
    # wei_sampler = WeightedRandomSampler(weights, 10, True)
    print('Reading training data...')
    train_dataset = Mydataset(config=config, data=train_df, with_labels=True)
    train_iter = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    print('Reading validation data...')
    valid_dataset = Mydataset(config=config, data=valid_df, with_labels=True)
    dev_iter = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=True)

    model = bert_RNN(config).to(config.device)
    train(config, model, train_iter, dev_iter)

