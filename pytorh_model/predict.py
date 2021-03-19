# coding: UTF-8
import time
import torch
import numpy as np
import json
import pandas as pd
import os
from train_eval import train, init_network, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from pytorch_pretrained_bert import BertTokenizer
from bert import bert_model
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/all_label_train_1.csv'  # 训练集
        self.dev_path = dataset + '/all_label_valid_1.csv'  # 训练集
        self.test_path = dataset + '/test_data.csv'  # 测试集
        self.submit_example_path = dataset + '/submit_example.csv'  # 提交格式
        self.new_submit_example_path = dataset + '/new_submit_example.csv'  # 提交格式
        self.id2label_path = dataset + '/id2label.json'  # label集合
        self.class_list, _ = json.load(open(self.id2label_path))  # 类别名单
        self.save_path = dataset + '/saved_dict_torch/' + self.model_name + '512-1.bin'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 100000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
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

    start_time = time.time()
    print("Loading data...")
    test_data = build_dataset(config, config.test_path, test_flag=True)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    model = bert_model(config).to(config.device)
    predict_all = test(config, model, test_iter)

    # ---------------------生成文件--------------------------
    df_test = pd.read_csv(config.submit_example_path, encoding='utf-8')
    id2label, label2id = json.load(open(config.id2label_path))
    id2label = {int(i): j for i, j in id2label.items()}  # 转为int型(原本是字符串形式)
    class_labels = []
    rank_labels = []
    for i in predict_all:
        label = str(id2label[i])
        class_labels.append(label)
        if label in ['财经', '时政']:
            rank_label = str('高风险')
        elif label in ['房产', '科技']:
            rank_label = str('中风险')
        elif label in ['教育', '时尚', '游戏']:
            rank_label = str('低风险')
        else:
            rank_label = str('可公开')
        rank_labels.append(rank_label)

    df_test['class_label'] = class_labels
    df_test['rank_label'] = rank_labels
    df_test.to_csv(config.new_submit_example_path, index=False, columns=['id', 'class_label', 'rank_label'])
