from sklearn.model_selection import StratifiedKFold  # 分层K折
import pandas as pd
import numpy as np


def split_csv(infile, train_file, valid_file, seed=999, ratio=0.2):
    infile_df = pd.read_csv(infile, encoding='utf-8')
    print(infile_df.head())
    print(infile_df.shape)
    skf = StratifiedKFold(n_splits=5, random_state=1314, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(infile_df['content'], infile_df['label_id']), 1):
        print(f'Fold {fold}')
        infile_df.iloc[train_idx, :].to_csv(train_file + f'_{fold}.csv', encoding='utf-8', index=False)
        infile_df.iloc[valid_idx, :].to_csv(valid_file + f'_{fold}.csv', encoding='utf-8', index=False)


split_csv(infile='all_label_id_data_3class.csv', train_file='all_label_train_3class', valid_file='all_label_valid_3class')
"""
test_file_path = 'test_data.csv'
test_df = pd.read_csv(test_file_path, encoding='utf-8')
test_df['label_id'] = 0
test_df.to_csv('new_test_data.csv', encoding='utf-8', index=False, columns=['id', 'label_id', 'content'])
"""