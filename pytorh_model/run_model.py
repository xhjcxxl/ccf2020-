# coding: UTF-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import time
import torch
import numpy as np
import json
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from pytorch_pretrained_bert import BertTokenizer
from bert import bert_model


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
        self.class_list, _ = json.load(open(self.id2label_path))  # 类别名单
        self.save_path = dataset + '/saved_dict_torch/' + self.model_name + '1231_3class_768_1.bin'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 100000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 5  # epoch数
        self.batch_size = 8  # mini-batch大小
        self.pad_size = 512  # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5  # 学习率
        self.bert_path = '../bert/torch_bert_chinese/'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        print(self.tokenizer, self.bert_path)
        self.hidden_size = 768
        self.seed = 1314


if __name__ == '__main__':
    dataset = '../dataset'  # 数据集
    config = Config(dataset)
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # train
    for i in range(1, 5):
        print("当前是: ", i)
        config.train_path = dataset + '/all_label_valid_3class_'+str(i)+'.csv'
        config.dev_path = dataset + '/all_label_valid_3class_'+str(i)+'.csv'
        config.save_path = dataset + '/saved_dict_torch/model_' + config.model_name + '1231_3class_768_'+str(i)

        start_time = time.time()
        print("Loading data...")
        train_data = build_dataset(config, config.train_path, test_flag=False)
        dev_data = build_dataset(config, config.dev_path, test_flag=False)

        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        model = bert_model(config).to(config.device)
        train(config, model, train_iter, dev_iter)
