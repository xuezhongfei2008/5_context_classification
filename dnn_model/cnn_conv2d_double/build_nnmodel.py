# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle


def load_embedding():
    # 读取word2vec词向量
    embedding_word2vec_array = np.load("../data/embedding_word2vec.npy")
    # embedding_fasttext_array = np.load("/opt/gongxf/python3_pj/Robot/4_cnn_simi/data/embedding_fasttext_cbow.npy")
    embedding_fasttext_array = np.load("../data/embedding_word2vec.npy")
    # 获取词汇大小
    vocab_word2vec_size = embedding_word2vec_array.shape[0]
    vocab_fasttext_size = embedding_fasttext_array.shape[0]
    # 回去词向量 shape
    embedding_dim = embedding_word2vec_array.shape[1]
    # 返回labels的种类数
    pickle_file = open('../data/categories.pkl', 'rb')
    categories = pickle.load(pickle_file)
    categories_len = len(categories)
    return embedding_word2vec_array, embedding_fasttext_array, vocab_word2vec_size, vocab_fasttext_size, embedding_dim, categories_len  # [0,1,2,3,4,5]


class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = load_embedding()[4]  # 词向量维度,使用gensim 设置的值
    seq_length = 8  # 序列长度
    num_classes = load_embedding()[5]  # 类别数
    vocab_word2vec_size = load_embedding()[2]  # 词汇表达小 使用gensim 设置的值
    vocab_fasttext_size = load_embedding()[3]
    vocab_size=vocab_word2vec_size
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    batch_size = 256  # 每批训练大小
    num_epochs = 20  # 总迭代轮次
    print_per_batch = 5  # 每多少轮输出一次结果
    save_per_batch = 100  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def input_embedding(self):
        # 词向量的转换# 词向量映射
        with tf.device('/cpu:0'):
            # 当trainable 值设为True时，该模型就是non_static,当trainable值设为False时模型就是static的
            embedding_word2vec = tf.get_variable('embedding_word2vec', trainable=True, initializer=load_embedding()[0])
            embedding_inputs_word2vec = tf.nn.embedding_lookup(embedding_word2vec, self.input_x)
            embedding_inputs_word2vec = tf.expand_dims(embedding_inputs_word2vec, -1)
            #fasttext词向量
            embedding_fasttext = tf.get_variable('embedding_fasttext', trainable=False, initializer=load_embedding()[0])
            embedding_inputs_fasttext = tf.nn.embedding_lookup(embedding_fasttext, self.input_x)
            embedding_inputs_tasttext = tf.expand_dims(embedding_inputs_fasttext, -1)
        return embedding_inputs_word2vec,embedding_inputs_tasttext

    def cnn(self):
        """CNN模型"""
        embedding_inputs_word2vec,embedding_inputs_tasttext = self.input_embedding()
        pooled_outputs = []
        filter_sizes = [[1, 150], [2, 150], [3, 100], [4, 100],[5, 100]]
        num_filters_total = 0
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cnn%s" % filter_size[0]):
                filter_shape = [filter_size[0], self.config.embedding_dim, 1, filter_size[1]]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W1')
                W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W2')
                #word2vec卷积层
                conv_word2vec = tf.nn.conv2d(
                    embedding_inputs_word2vec,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_word2vec")
                pooled_word2vec = tf.nn.max_pool(
                    conv_word2vec,
                    ksize=[1, self.config.seq_length - filter_size[0] + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_word2vec")

                # fasttext卷积层
                conv_fasttext = tf.nn.conv2d(
                    embedding_inputs_tasttext,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_fasttext")
                pooled_fasttext = tf.nn.max_pool(
                    conv_fasttext,
                    ksize=[1, self.config.seq_length - filter_size[0] + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_fasttext")
                pooled_outputs.append(pooled_word2vec)
                pooled_outputs.append(pooled_fasttext)
            num_filters_total += filter_size[1]
        self.h_pool = tf.concat(pooled_outputs, -1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total*2])  # 映射成一个300维的特征向量
        print("self.h_pool_flat", self.h_pool_flat.shape)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # 隐藏层函数定义
            cur_layer = self.h_pool_flat
            layer_dimension = [250,125] # 神经网络层节点的个数
            n_layers = len(layer_dimension)  # 神经网络的层数
            for i in range(0, n_layers):
                out_dimension = layer_dimension[i]
                fc = tf.layers.dense(inputs=cur_layer, \
                                     units=out_dimension, \
                                     activation=tf.nn.relu \
                                     # activity_regularizer=tf.contrib.layers.l2_regularizer(0.001), \
                                     # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),\
                                     # name='fc1'
                                     )
                cur_layer = tf.contrib.layers.dropout(fc, self.keep_prob)

            # 输出层,分类器
            self.logits = tf.layers.dense(cur_layer, self.config.num_classes, name='fc2')
            print("self.logits", self.logits.shape)
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            print("self.y_pred", self.y_pred.shape)

        with tf.name_scope("loss"):
            # 使用优化方式，损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
        with tf.name_scope("optimize"):
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.y_pred, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")


class TRNNConfig(object):
    """CNN配置参数"""
    embedding_dim = load_embedding()[2]  # 词向量维度,使用gensim 设置的值
    seq_length = 10  # 序列长度
    num_classes = load_embedding()[3]  # 类别数
    vocab_size = load_embedding()[1]  # 词汇表达小 使用gensim 设置的值
    num_layers = 2  # 隐藏层数
    hidden_dim = 100  # 全连接层神经元
    rnn = 'gru'  # lstm 或者 gru
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextRNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn()

    def input_embedding(self):
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', initializer=load_embedding()[0])
            # embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],initializer=load_embedding()[0])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return embedding_inputs

    def rnn(self):
        """CNN模型"""
        embedding_inputs = self.input_embedding()

        def lstm_cell():
            # lstm核
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        # gru核
        def gru_cell():
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)

        # 定义dropout
        def dropout():
            # 添加dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.name_scope("rnn"):
            # RNN layer
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # 定义全连接层，隐藏层函数定义
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            # 加入keep_prob特性
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # 使用激活函数
            fc = tf.nn.relu(fc)

            # 分类器，输出层
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("loss"):
            # 使用优化方式，损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
        with tf.name_scope("optimize"):
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
