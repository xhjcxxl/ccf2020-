import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy
from collections import Counter
from matplotlib.font_manager import FontProperties

# 数据探索
df_train = pd.read_csv('../dataset/old/train_set.csv', sep='\t')
df_test = pd.read_csv('../dataset/old/test_a.csv', sep='\t')

# 简单查看数据
print(df_train.head())
print(len(df_train))
"""
   label                                               text
0      2  2967 6758 339 2021 1854 3731 4109 3792 4149 15...
1     11  4464 486 6352 5619 2465 4802 1452 3137 5778 54...
2      3  7346 4068 5074 3747 5681 6093 1777 2226 7354 6...
3      2  7159 948 4866 2109 5520 2490 211 3956 5520 549...
4      3  3646 3055 3055 2490 4659 6065 3370 5814 2465 5...
200000
"""

# 发现text域的数据是字符串。我们想要得到整数序列。可以用字符串分割split()
print(len(df_train['text'][0]), type(df_train['text'][0]))
# 5120 <class 'str'>


# 查看数据长度分布，从而知道从哪里截断，取多长合适

# 当前使用的函数split_df负责将一行数据按空格切分成整数列表，
# 然后计算该列表的长度。
def split_df(df_row):
    return len(str(df_row).split())


len_dist = np.vectorize(split_df)(df_train['text'])
len_test_dist = np.vectorize(split_df)(df_test['text'])

# 使用describe函数查看训练集和测试集中的数据长度分布
print(pd.Series(len_dist).describe())
print(pd.Series(len_test_dist).describe())
"""
count    200000.000000
mean        907.207110
std         996.029036
min           2.000000
25%         374.000000
50%         676.000000
75%        1131.000000
max       57921.000000

dtype: float64
count    50000.000000
mean       909.844960
std       1032.313375
min         14.000000
25%        370.000000
50%        676.000000
75%       1133.000000
max      41861.000000
dtype: float64
"""


# 直方图
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax = plt.hist(x=len_dist, bins=100)
ax = plt.hist(x=len_test_dist, bins=100)

plt.xlim([0, max(max(len_dist), max(len_test_dist))])
plt.xlabel("length of sample")
plt.ylabel("number of sample")
plt.legend('train_len', 'test_len')
plt.show()

# 使用seaborn绘制更好的图。seaborn计算的纵坐标是频率，而不是出现次数。
# 由于训练集和测试集的数据量不一样，因此用频率更加科学、更能看出是否符合同一分布。
plt.figure(figsize=(15, 5))
ax = sns.distplot(len_dist, bins=100)
ax = sns.distplot(len_test_dist, bins=100)
plt.xlim([0, max(max(len_dist), max(len_test_dist))])
plt.xlabel("length of sample")
plt.ylabel("number of sample")
plt.legend('train_len', 'test_len')
plt.show()

# 同分布验证
# 检验指定的两个数列是否服从相同分布
"""
使用ks_2samp检验两个数列是否来自同一个样本，提出假设：len_dist和len_test_dist服从相同的分布。
Ks_2sampResult(statistic=0.004049999999999998, pvalue=0.5279614323123156)
最终返回的结果，pvalue=0.5279614323123156，比指定的显著水平（假设为5%）大，则我们完全可以接受假设：len_dist和len_test_dist服从相同的分布。

"""
print(scipy.stats.ks_2samp(len_dist, len_test_dist))
# Ks_2sampResult(statistic=0.004049999999999998, pvalue=0.5279614323123156)
# 截断位置
"""
考虑到数据长度分布是长尾分布，log一下看看是不是`正态分布`，如果是`正态分布`，使用`3sigma`法则作为截断的参考。如果不是，则就只能瞎猜了

测量拟合分布的均值和方差sigma原则：
* `1σ原则`：数值分布在`(μ-σ,μ+σ)`中的概率为`0.6526`；
* `2σ原则`：数值分布在`(μ-2σ,μ+2σ)`中的概率为`0.9544`；
* `3σ原则`：数值分布在`(μ-3σ,μ+3σ)`中的概率为`0.9974`；

由于“小概率事件”和假设检验的基本思想 “小概率事件”通常指发生的概率小于5%的事件，认为在一次试验中该事件是几乎不可能发生的。
由此可见X落在(μ-3σ,μ+3σ)以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，基本上可以把区间(μ-3σ,μ+3σ)看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。
"""
log_len_dist = np.log(1 + len_dist)  # 取对数
log_len_test_dist = np.log(1 + len_test_dist)
plt.figure(figsize=(15, 5))
ax = sns.distplot(log_len_dist)  # 直接生成图
ax = sns.distplot(log_len_test_dist)
plt.xlabel("log length of sample")
plt.ylabel("prob of log")
plt.legend(['train_len', 'test_len'])
plt.show()

# 先验证训练集分布是否为正态分布：
# scipy.stats.kstest 判断分布是否为正太分布
_, lognormal_ks_pvalue = scipy.stats.kstest(rvs=log_len_dist, cdf='norm')
print(lognormal_ks_pvalue)
# 0.0 拟合优度检验，p值为0，意思就是说这不是一个正态分布

# 之前我们把数据log了一下，但是这里有更科学的变换方式。
# log只是box-cox变换的特殊形式。我们使用box-cox变换再次做一下验证，是否为正态分布：
trans_data, lam = scipy.stats.boxcox(len_dist + 1)
print(scipy.stats.normaltest(trans_data))
# NormaltestResult(statistic=1347.793358118494, pvalue=2.1398873511704724e-293)
# p值约等于0，这说明我们的假设不成立。
# 但总归是要猜一个截断值的。看log图上8.5的位置比较靠谱。np.exp(8.5)=4914约等于5000，因此我初步决定把截断长度定为5000。


# 简单查看类别信息表
"""
先改造一下df_train，多加几个字段，分别是：
* text-split，将text字段分词
* len，每条新闻长度
* first_char，新闻第一个字符
* last_char，新闻最后一个字符
* most_freq，新闻最常出现的字符
"""
df_train['text_split'] = df_train['text'].apply(lambda x: x.split())
df_train['len'] = df_train['text'].apply(lambda x: len(x.split()))
df_train['first_char'] = df_train['text_split'].apply(lambda x: x[0])
df_train['last_char'] = df_train['text_split'].apply(lambda x: x[-1])
df_train['most_freq'] = df_train['text_split'].apply(lambda x: np.argmax(np.bincount(x)))
print(df_train.head())
"""
   label  ... most_freq
0      2  ...      3750
1     11  ...      3750
2      3  ...      3750
3      2  ...      3750
4      3  ...      3055
"""


# 构建一个类别信息表
"""
构建一个类别信息表。
* count，该类别新闻个数
* len_mean，该类别新闻平均长度
* len_std，该类别新闻长度标准差
* len_min，该类别新闻长度最小值
* len_max，该类别新闻长度最大值
* freq_fc，该类别新闻最常出现的第一个字符
* freq_lc，该类别新闻最常出现的最后一个字符
* freq_freq，该类别新闻最常出现的字符
"""
df_train_info = pd.DataFrame(columns=['count', 'len_mean', 'len_std', 'len_min',
                                      'len_max', 'freq_fc', 'freq_lc', 'freq_freq'])
for name, group in df_train.groupby('label'):
    # 对 属性 ‘label’ 进行统计分析
    count = len(group)  # 该类别新闻数
    len_mean = np.mean(group['len'])  # 该类别长度平均值
    len_std = np.std(group['len'])  # 长度标准差
    len_min = np.min(group['len'])  # 最短的新闻长度
    len_max = np.max(group['len'])  # 最长的新闻长度
    freq_fc = np.argmax(np.bincount(group['first_char']))  # 最频繁出现的首词
    freq_lc = np.argmax(np.bincount(group['last_char']))  # 最频繁出现的末词
    freq_freq = np.argmax(np.bincount(group['most_freq']))  # 该类别最频繁出现的词
    df_train_info.loc[name] = [count, len_mean, len_std, len_min,
                               len_max, freq_fc, freq_lc, freq_freq]
print(df_train_info)
"""
      count     len_mean      len_std  ...  freq_fc  freq_lc  freq_freq
0   38918.0   878.717663   859.302990  ...   2400.0    900.0     3750.0
1   36945.0   870.363676  1451.060541  ...   1141.0    900.0     3750.0
2   31425.0  1014.429562   737.313693  ...   1580.0   2662.0     3750.0
3   22133.0   784.774726   739.347231  ...   7346.0    900.0     3750.0
4   15016.0   649.705647   718.689556  ...   1141.0    900.0     3750.0
5   12232.0  1116.054938   910.002484  ...   5744.0    900.0     3750.0
6    9985.0  1249.114071  1203.464887  ...   3659.0    900.0     3750.0
7    8841.0  1157.883271   942.048602  ...   6835.0    900.0     3750.0
8    7847.0   712.401172   898.704321  ...    913.0    900.0     3750.0
9    5878.0   833.627084   739.593276  ...   7346.0    900.0     3750.0
10   4920.0   911.138008   958.311796  ...   3523.0    885.0     3750.0
11   3131.0   608.889812   509.755296  ...   6811.0    900.0     3750.0
12   1821.0  1194.969248  1108.697967  ...   5006.0    900.0     3750.0
13    908.0   735.325991   795.676666  ...   1999.0   2662.0     3750.0
"""


# 类别分布
# zhfont = FontProperties(size=14) 字体设置
label_2_index_dict = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5,
                      '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11,
                      '彩票': 12, '星座': 13}
index_2_label_dict = {v: k for k, v in label_2_index_dict.items()}

plt.figure()
plt.bar(x=range(14), height=np.bincount(df_train['label']))
plt.xlabel("label")
plt.ylabel("number of sample")
plt.xticks(range(14), list(index_2_label_dict.values()), rotation=60)
plt.show()
"""
从统计结果可以看出
* 赛题的数据集类别分布存在较为不均匀的情况。在训练集中科技类新闻最多，其次是股票类新闻，最少的新闻是星座新闻。
* 由于类别不均衡，会严重影响模型的精度。但是我们也是有办法应对的。
"""

# 类别长度
df_train['len'] = df_train['text'].apply(lambda x: len(x.split()))
plt.figure()
ax = sns.catplot(x='label', y='len', data=df_train, kind='strip')
plt.xticks(range(14), list(index_2_label_dict.values()), rotation=60)
plt.show()
"""
不同类别的文章长度不同，可以把长度作为一个Feature，以供机器学习模型训练
"""

# 字符分布
# 训练集中总共包括6869个字，最大数字为7549，最小数字为0，
# 其中编号3750的字出现的次数最多，编号3133的字出现的次数最少，仅出现一次。
all_lines = ' '.join(list(df_train['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
print(len(word_count))
# 6869
print(word_count[0])
# ('3750', 7482224)
print(word_count[-1])
# ('3133', 1)

"""
下面代码统计了不同字符在多少个句子中出现过，
其中字符3750、字符900和字符648在20w新闻的覆盖率接近99%，很有可能是标点符号。
"""
df_train['text_unique'] = df_train['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(df_train['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: int(d[1]), reverse=True)
# 打印整个训练集中覆盖率前5的词
for i in range(5):
    print("{} occurs {} times, {}%".format(word_count[i][0], word_count[i][1], (word_count[i][1] / 200000) * 100))
"""
3750 occurs 197997 times, 98.9985%
900 occurs 197653 times, 98.8265%
648 occurs 191975 times, 95.9875%
2465 occurs 177310 times, 88.655%
6122 occurs 176543 times, 88.2715%
"""

# 分析总结
"""
1. 训练集共200,000条新闻，每条新闻平均907个字符，最短的句子长度为2，最长的句子长度为57921，其中75%以下的数据长度在1131以下。
    测试集共50,000条新闻，每条新闻平均909个字符，最短句子长度为14，最长句子41861,75%以下的数据长度在1133以下。
    
2. 训练集和测试集就长度来说似乎是同一分布，但是不属于正态分布。

3. 赛题的数据集类别分布存在较为不均匀的情况。在训练集中科技类新闻最多，其次是股票类新闻，最少的新闻是星座新闻。需要用采样方法解决。
    文章最长的是股票类新闻。不同类别的文章长度不同，可以把长度和句子个数作为一个Feature，以供机器学习模型训练。

4. 训练集中总共包括6869个字，最大数字为7549，最小数字为0，其中编号3750的字出现的次数最多，编号3133的字出现的次数最少，仅出现一次。
    其中字符3750、字符900和字符648在20w新闻的覆盖率接近99%，很有可能是标点符号。

5. 900很有可能是句号，2662和885则很有可能为感叹号和问号，3750出现频率很高但是基本不在新闻最后出现，因此初步判断为逗号。
    按照这种划分，训练集中每条新闻平均句子个数约为19。

6. 在训练集中，不同类别新闻出现词汇有特色。但是需要把共有的常用词停用。自然想到利用TF-IDF编码方式。
"""
