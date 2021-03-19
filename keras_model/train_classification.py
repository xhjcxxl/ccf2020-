# 分类模型训练
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 设置GPU编号
import tensorflow as tf
import numpy as np
import pandas as pd
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.callbacks import ReduceLROnPlateau
from keras.layers import *
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold  # 分层K折
from keras.utils import multi_gpu_model
from multi_gpu import to_multi_gpu


# bert config(这里使用rank1的训练好的，自己训练太麻烦了，不过能跑通，能理解)
config_path = 'bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'bert/chinese_L-12_H-768_A-12/vocab.txt'

n = 5  # cross-validation
seed = 2020
num_classes = 10

maxlen = 512
max_segment = 2  # 设定的多大segment
batch_size = 4
# batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
grad_accum_steps = 64  # 梯度积累，即积累一定梯度后再进行运算
drop = 0.2  # dropout
lr = 2e-5
epochs = 3


def load_data(df):
    """    加载数据    """
    D = list()
    for _, row in df.iterrows():  # 按行读取
        text = row['content']
        label = row['label_id']
        D.append((text, int(label)))
        # D = [(text1, label1), (text2, label2), ...]
    return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def sentence_split(words):
    """ 句子截断 """
    #  此次检测出来文本长度平均在1020 mean：1074.372215 std：1090.884148左右，故取1530
    document_len = len(words)  # 文本总长度
    # [0, 510, 1020, 1530, 2040, 2550, 3060, 3570, 4080, 4590]
    # 为文档 按照maxlen 划分后的 索引 位置（没有最后部分的位置，即不足maxlen的那段）
    index = list(range(0, document_len, maxlen-2))
    index.append(document_len)  # 加上最后的位置

    segments = []
    for i in range(len(index) - 1):
        # 这是标准长度 maxlen-2的文本，因为一个段落太长，所以需要这样截断才能训练
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        # 转化为id, 并加上 首尾的cls和sep
        segment = tokenizer.tokens_to_ids(['[CLS]'] + segment + ['[SEP]'])
        segments.append(segment)

    assert len(segments) > 0
    # 对划分的段进行判断，设定不超过两个，因为Bert输入就是不超过两个
    # 如果超过两个段
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        # 只取开头一段和结尾一段，一共两段，满足max_segment要求
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments


class data_generator(DataGenerator):
    """ 数据生成器 """
    def __init__(self, data, batch_size=32, buffer_size=None, random=False):
        super().__init__(data, batch_size, buffer_size)
        self.random = random

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids = sentence_split(text)  # 句子截断
            token_ids = sequence_padding(token_ids, length=maxlen)  # padding
            segment_ids = np.zeros_like(token_ids)  # 获取与token_ids维度一样的全零数据

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            # 所有的ids 都进行padding
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(
                    batch_token_ids, length=max_segment
                )
                batch_segment_ids = sequence_padding(
                    batch_segment_ids, length=max_segment
                )
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d


class Attention(Layer):
    """ 注意力层 """
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.truncated_normal(mean=0.0, stddev=0.05)  # 初始化为正态分布
        # 为该层创建一个可训练的权重
        self.weight = self.add_weight(
            name='weight',
            shape=(self.hidden_size, self.hidden_size),
            initializer=initializer,
            trainable=True
        )
        # 为该层创建一个可训练的权重
        self.bias = self.add_weight(
            name='bias',
            shape=(self.hidden_size,),
            initializer='zero',
            trainable=True
        )
        # 为该层创建一个可训练的权重
        self.query = self.add_weight(
            name='query',
            shape=(self.hidden_size, 1),
            initializer=initializer,
            trainable=True
        )

        super().build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        x, mask = x
        # 因为 self.weight只有两个维度，所以这里要进行维度处理
        mask = K.squeeze(mask, axis=2)  # 维度压缩 去掉一个维度 axis=2,但是数据还是不变的
        # linear 线性变化
        # K.dot()进行 点乘，然后加了self.bias
        key = K.bias_add(K.dot(x, self.weight), self.bias)

        # compute attention
        outputs = K.squeeze(K.dot(key, self.query), axis=2)  # 计算注意力
        outputs -= 1e32 * (1 - mask)

        attn_scores = K.softmax(outputs)  # 使用 softmax 计算得分
        attn_scores *= mask
        attn_scores = K.reshape(
            attn_scores, shape=(-1, 1, attn_scores.shape[-1])
        )

        outputs = K.squeeze(K.batch_dot(attn_scores, key), axis=1)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.hidden_size


def build_model():
    """ 模型构建 """
    token_ids = Input(shape=(max_segment, maxlen), dtype='int32')
    segment_ids = Input(shape=(max_segment, maxlen), dtype='int32')

    input_mask = Masking(mask_value=0)(token_ids)  # 对输入token_ids做masking
    # k.any()先归约，然后再进行类型变换
    # 可以转换一个 Keras 变量，但它仍然返回一个 Keras 张量（类型变换）
    input_mask = Lambda(
        lambda x: K.cast(K.any(x, axis=2, keepdims=True), 'float32')
    )(input_mask)

    # 重构 维度 把 batch, token_ids 合并成一个维度
    token_ids1 = Lambda(
        lambda x: K.reshape(x, shape=(-1, maxlen))
    )(token_ids)
    segment_ids1 = Lambda(
        lambda x: K.reshape(x, shape=(-1, maxlen))
    )(segment_ids)

    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
    output = bert.model([token_ids1, segment_ids1])
    output = Lambda(lambda x: x[:, 0])(output)  # 取CLS 只取第一列
    # 维度重构
    output = Lambda(
        lambda x: K.reshape(x, shape=(-1, max_segment, output.shape[-1]))
    )(output)
    output = Multiply()([output, input_mask])  # 把输出和 input_mask拼到一起，然后输出一个张量，维度不变
    output = Dropout(drop)(output)

    output = Attention(output.shape[-1].value)([output, input_mask])  # 使用注意力
    output = Dropout(drop)(output)
    # FC 线性层
    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model([token_ids, segment_ids], output)
    # 设置多GPU
    # 设置优化器，优化参数
    optimizer_params = {
        'learning_rate': lr,
        'grad_accum_steps': grad_accum_steps
    }

    optimizer = extend_with_gradient_accumulation(Adam)  # 加入梯度累积
    optimizer = optimizer(**optimizer_params)

    # multi gpu

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def adversarial_training(model, embedding_name, epsilon=1.):
    """ 给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding 梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为 dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层

    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


class Evaluator(Callback):
    def __init__(self, valid_generator):
        super().__init__()
        self.valid_generator = valid_generator  # 验证集数据生成器
        self.best_val_f1 = 0.

    def evaluate(self):
        y_true, y_pred = list(), list()
        for x, y in self.valid_generator:
            y_true.append(y)  # 真实结果
            y_pred.append(self.model.predict(x).argmax(axis=1))  # 预测结果
        y_true = np.concatenate(y_true)  # 所有结果拼接在一起
        y_pred = np.concatenate(y_pred)  # 所有结果拼接在一起
        f1 = f1_score(y_true, y_pred, average='macro')  # 计算f1值
        return f1

    def on_epoch_end(self, epoch, logs=None):  # 每个epoch结束时，都会执行这个
        val_f1 = self.evaluate()  # 获取f1
        if val_f1 > self.best_val_f1:  # 如果这轮epoch的f1 比历史最佳的f1高，就替换掉
            self.best_val_f1 = val_f1
        logs['val_f1'] = val_f1
        print(f'val_f1: {val_f1:.5f}, best_val_f1: {self.best_val_f1:.5f}')


# 执行训练
def do_train(df_train):
    # n 折
    skf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)  # 设置 n折
    # skf.split(df_train['text'], df_train['label'])  划分数据，生成 train, valid数据
    # enumerate(data, 1) 表示下标从1开始，即 fold从1开始计算
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df_train['content'], df_train['label_id']), 1):
        print(f'Fold {fold}')
        # 加载数据
        train_data = load_data(df_train.iloc[train_idx])
        valid_data = load_data(df_train.iloc[valid_idx])
        # 加入数据迭代器中
        train_generator = data_generator(train_data, batch_size, random=True)
        valid_generator = data_generator(valid_data, batch_size)

        model = build_model()  # 构建模型
        # strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
        # with strategy.scope():
        #     model = build_model()
        #     model.summary()

        # 加入对抗训练
        adversarial_training(model, 'Embedding-Token', 0.5)  # 加入对抗训练
        # 回调函数
        callbacks = [
            Evaluator(valid_generator),  # 每个epoch结束时，就会执行验证
            EarlyStopping(
                monitor='val_f1',
                patience=5,
                verbose=1,
                mode='max'),  # 早期停止条件，监控val_f1值，如果5次都没有超过最佳f1，那么就停止训练
            ReduceLROnPlateau(
                monitor='val_f1',
                factor=0.5,
                patience=2,
                verbose=1,
                mode='max'),  # 当训练的模型停止提升的时候，就减少学习率，看是否能够继续提升
            ModelCheckpoint(
                f'weights-{fold}.h5',  # 保存路径 避免文件名被覆盖
                monitor='val_f1',
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
                mode='max'),  # 模型检查点，进行模型的数据进行保存；只保存最新f1的那次数据，只保存权重
        ]
        # 模型训练
        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=valid_generator.forfit(),
            validation_steps=len(valid_generator)
        )

        del model  # 删除模型
        K.clear_session()  # 清理 会话


if __name__ == '__main__':
    df_train = pd.read_csv('dataset/all_label_id_data.csv', encoding='utf-8')
    df_train['content'] = df_train['content'].apply(lambda x: x.strip().split())

    do_train(df_train)
