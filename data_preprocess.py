import numpy as np
import pandas as pd
import json
import codecs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy
from collections import Counter
label_data_path = 'dataset/labeled_data.csv'
unlabel_data_path = 'dataset/unlabeled_data.csv'
new_unlabel_data_path = 'dataset/new_unlabeled_data.csv'
label_seeds_path = 'dataset/label_seeds.json'
all_label_data_path = 'dataset/all_label_data.csv'

all_label_data = pd.read_csv(all_label_data_path, encoding='utf-8')

label2id = {'财经': 0, '房产': 1, '家居': 2, '教育': 3, '科技': 4, '时尚': 5, '时政': 6, '游戏': 7, '娱乐': 8, '体育': 9}
id2label = {v: k for k, v in label2id.items()}

all_label_data['label'] = 11
for i in range(all_label_data.shape[0]):
    # print(all_label_data['class_label'][i], label2id[all_label_data['class_label'][i]])
    all_label_data['label'] = label2id[all_label_data['class_label'][i]]
print(all_label_data.head)

plt.figure()
plt.bar(x=range(10), height=np.bincount(all_label_data['label']))
plt.xlabel("label")
plt.ylabel("number of sample")
plt.xticks(range(10), list(id2label.values()), rotation=60)
plt.show()