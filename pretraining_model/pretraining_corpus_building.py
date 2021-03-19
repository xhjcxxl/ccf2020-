# 构建预训练语料库
import glob
import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import numpy as np
import pandas as pd
import tensorflow as tf
from bert4keras.backend import K
from bert4keras.snippets import parallel_apply
from bert4keras.tokenizers import Tokenizer
from tqdm import tqdm


class TrainingDataset(object):
    """
    预训练数据集生成器
    """

    def __init__(self, tokenizer, sequence_length=512):
        """
        参数说明：tokenizer必须是bert4keras自带的tokenizer类
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id  # 设定 padding id
        self.token_cls_id = tokenizer._token_start_id  # 设定 CLS id
        self.token_sep_id = tokenizer._token_end_id  # 设定 SEP id
        self.token_mask_id = tokenizer._token_mask_id  # 设定 mask id
        self.vocab_size = tokenizer._vocab_size  # 获取 词表 大小

    def padding(self, sequence, padding_value=None):
        """ 对单个序列进行补0 """
        if padding_value is None:  # 没有指定填充内容，就使用默认的填充内容
            padding_value = self.token_pad_id
        sequence = sequence[:self.sequence_length]  # 多了截断，少了就少了
        padding_length = self.sequence_length - len(sequence)  # 不够的部分有多少
        # 原序列 + 补0的部分，总共的长度是不变了，都统一成sequence_length
        return sequence + [padding_value] * padding_length

    def sentence_process(self, text):
        """
        单个文本的处理函数，返回处理后的instance
        分词，转id,以及mask处理
        返回：[token_ids, mask_ids]
        """
        raise NotImplementedError

    def paragraph_process(self, texts, starts, ends, paddings):
        """
        单个段落（多个文本）处理函数
        :param texts: 单据组成的list
        :param starts: 每个instance的起始id
        :param ends: 每个instance的终止id
        :param paddings: 每个instance需要填充的id
        做法：不断塞句子，直到长度最接近sequence_length，然后padding
        """
        instances, instance = [], [[start] for start in starts]
        for text in texts:
            # 处理单个句子
            sub_instance = self.sentence_process(text)
            """
            sub_instance = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 1, 1, 1, 1, 1, 0, 0]]
            sub_instance = [i[:8 - 2] for i in sub_instance]
            print(sub_instance)
            [[1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1]]
            """
            # -2的原因是要留出来放 CLS和SEP
            sub_instance = [i[:self.sequence_length - 2] for i in sub_instance]
            new_length = len(instance[0]) + len(sub_instance[0])

            # 如果长度即将溢出
            if new_length > self.sequence_length - 1:
                # 插入终止符，并padding
                complete_instance = []
                for item, end, pad in zip(instance, ends, paddings):
                    item.append(end)  # 终止符
                    item = self.padding(item, pad)  # padding
                    complete_instance.append(item)  # 修改后的instance

                # 存储结果，并构建新样本
                instances.append(complete_instance)
                instance = [[start] for start in starts]

            # 样本续接
            for item, sub_item in zip(instance, sub_instance):
                item.extend(sub_item)

        # 插入终止符，并padding
        complete_instance = []
        for item, end, pad in zip(instance, ends, paddings):
            item.append(end)
            item = self.padding(item, pad)
            complete_instance.append(item)

        # 存储最后的instance
        instances.append(complete_instance)

        return instances

    def tfrecord_serialize(self, instances, instance_keys):
        """
        （序列化）转为tfrecord的字符串，等待写到文件中去
        """

        def create_feature(x):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

        serialized_instances = []
        for instance in instances:
            features = {
                k: create_feature(v) for k, v in zip(instance_keys, instance)
            }
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            serialized_instance = tf_example.SerializeToString()
            serialized_instances.append(serialized_instance)

        return serialized_instances

    def process(self, corpus, record_name, workers=8, max_queue_size=2000):
        """
        处理函数（主要调用的就是这个函数）
        处理输入语料（corpus）,最终转为tfrecord格式(record_name)，生成对应的文件
        自带多进程支持，如果cpu核心数多，请加入workers和max_queue_size.
        """
        # 创建一个TFRecordWriter对象,这个对象(writer)就负责写记录到指定的文件中去了.
        # TFRecordWriter把记录写入到TFRecords文件的类.
        writer = tf.io.TFRecordWriter(record_name)

        globals()['count'] = 0

        def write_to_tfrecord(serialized_instances):
            globals()['count'] += len(serialized_instances)
            for serialized_instance in serialized_instances:
                writer.write(serialized_instance)  # 写入到文件中

        def paragraph_process(texts):
            instances = self.paragraph_process(texts)  # 段落处理
            serialized_instances = self.tfrecord_serialize(instances)  # 文本序列化（转化为tfrecord的字符串）
            return serialized_instances

        # 多进程/多线程处理
        parallel_apply(
            func=paragraph_process,
            iterable=corpus,
            workers=workers,
            max_queue_size=max_queue_size,
            callback=write_to_tfrecord,
        )
        writer.close()  # 关闭对象.
        print('write %s examples into %s' % (globals()['count'], record_name))

    @staticmethod
    def load_tfrecord(record_name, batch_size, parse_function):
        """
        加载处理成 tfrecord 格式的语料
        """
        if not isinstance(record_name, list):
            # 名字转成列表格式
            record_name = [record_name]

        dataset = tf.data.TFRecordDataset(record_name)  # 加载
        dataset = dataset.map(parse_function)  # 解析
        dataset = dataset.repeat()  # 循环
        dataset = dataset.shuffle(batch_size * 1000)  # 打乱
        dataset = dataset.batch(batch_size)  # 成批

        return dataset


class TrainingDatasetRoBERTa(TrainingDataset):
    """预训练数据集生成器（RoBERTa模式）"""

    def __init__(self, tokenizer, word_segment,
                 mask_rate=0.15, sequence_length=512):
        """
        tokenizer 必须是bert4keras自带的tokenizer类；
        word_segment 是任意分词函数。
        """
        super(TrainingDatasetRoBERTa, self).__init__(tokenizer, sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate

    def token_process(self, token_id):
        """
        以80%的几率替换为[MASK]，以10%的几率保持不变，以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def sentence_process(self, text):
        """
        单个文本的处理函数（将原本类里面的函数实现了）
        流程：
            1.分词
            2.转id
            3.按照mask_rate构建全词mask的序列来指定哪些token是否要被mask.
        """
        words = self.word_segment(text)  # 分词函数
        rands = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(text=word)[1:-1]  # 去掉了首尾的CLS和SEP
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)  # 转为id
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)

            mask_ids.extend(word_mask_ids)

        return [token_ids, mask_ids]

    def paragraph_process(self, texts):
        """ 给原方法补上starts, ends, paddings """
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super().paragraph_process(texts, starts, ends, paddings)

    def tfrecord_serialize(self, instances):
        """
        给原方法补上instance_keys
        """
        instance_keys = ['token_ids', 'mask_ids']
        return super().tfrecord_serialize(instances, instance_keys)

    @staticmethod
    def load_tfrecord(record_name, sequence_length, batch_size):
        """
        给原方法补上parse_function。
        parse_function: 解析函数
        用于解析序列化的数据，即 加载tfrecord字符格式的数据，要解析出来
        """

        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
                'mask_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            is_masked = K.not_equal(mask_ids, 0)
            masked_token_ids = K.switch(is_masked, mask_ids - 1, token_ids)
            """
            Input-Token：直接输入到Bert模型的
            Input-Segment：输入到Bert模型的
            """
            x = {
                'Input-Token': masked_token_ids,
                'Input-Segment': segment_ids,
                'token_ids': token_ids,
                'is_masked': K.cast(is_masked, K.floatx()),
            }
            y = {
                'mlm_loss': K.zeros([1]),
                'mlm_acc': K.zeros([1]),
            }
            return x, y

        return TrainingDataset.load_tfrecord(
            record_name, batch_size, parse_function
        )


if __name__ == '__main__':
    sequence_length = 512
    workers = 8  # 8个工作队列并行
    max_queue_size = 10000
    # 这个词表，因为是脱敏数据，只需要找到最大的数字，自己生成就行了
    # 如果不是脱敏数据，就可以直接使用原来的 bert预训练模型了，也就不需要进行语料库训练了
    dict_path = '../pre_models/vocab.txt'

    tokenizer = Tokenizer(dict_path, do_lower_case=True)


    def some_texts():
        # 使用训练集和测试集一共25万数据对Bert-base进行预训练
        # 所以这里就是直接使用 训练集和 测试集 的数据来进行预训练的
        filenames = glob.glob('../dataset/*')
        np.random.shuffle(filenames)  # 打乱文件
        count, texts = 0, []
        for filename in filenames:
            df = pd.read_csv(filename, sep='\t')  # 读取corpus文件
            for _, row in df.iterrows():
                l = row['text'].strip()
                # 如果 l 的长度 大于 sequence_length
                # 表示这个 text 很长很长，需要划分，就按照 sequence_length 划分
                # 然后整个文本 拼到一起
                if len(l.split()) > sequence_length - 2:
                    l = l.split()
                    len_ = sequence_length - 2
                    texts.extend([
                        ' '.join(l[i * len_: (i + 1) * len_])
                        for i in range((len(l) // len_) + 1)
                    ])
                else:
                    texts.extend([l])
                count += 1
                if count == 10:  # 10篇文章合在一起再处理
                    yield texts
                    count, texts = 0, []
        if texts:
            yield texts

    # 分词函数，在这个语料库中，因为数据脱敏了，所以直接可以按照空格分开了
    # 如果 是中文文本的话，就需要使用分词工具把它分开
    def word_segment(text):
        return text.split()

    TD = TrainingDatasetRoBERTa(
        tokenizer, word_segment, sequence_length=sequence_length
    )

    for i in range(10):  # 数据重复10遍
        TD.process(
            corpus=tqdm(some_texts()),
            record_name=f'../corpus_tfrecord/corpus.{i}.tfrecord',
            workers=workers,
            max_queue_size=max_queue_size,
        )
