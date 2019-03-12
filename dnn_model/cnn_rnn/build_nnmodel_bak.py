# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import ResidualWrapper
from tensorflow import layers

base_path = '/opt/gongxf/python3_pj/nlp_practice/5_context_classification/dnn_model'


def load_embedding():
    # 读取word2vec词向量
    embedding_array = np.load(base_path + "/data/embedding_word2vec.npy")
    # 获取词汇大小
    vocab_size = embedding_array.shape[0]
    # 回去词向量 shape
    embedding_dim = embedding_array.shape[1]
    # 返回labels的种类数
    pickle_file = open(base_path + '/data/categories.pkl', 'rb')
    categories = pickle.load(pickle_file)
    categories_len = len(categories)
    return embedding_array, vocab_size, embedding_dim, categories_len


class TRNNConfig(object):
    """CNN配置参数"""
    # 数据预处理参数
    embedding_dim = load_embedding()[2]  # 词向量维度,使用gensim 设置的值
    seq_length = 10  # 序列长度
    num_classes = load_embedding()[3]  # 类别数
    vocab_size = load_embedding()[1]  # 词汇表达小 使用gensim 设置的值

    # 模型构建参数
    num_layers = 2  # 隐藏层数
    hidden_dim = 200  # 全连接层神经元
    cell_type = 'lstm'  # lstm 或者 gru
    use_residual = True
    use_dropout = True
    time_major = False
    bidirectional = True
    dropout_keep_prob = 0.5  # dropout保留比例

    # 模型优化参数
    optimizer = 'adam'
    learning_rate = 1e-3
    min_learning_rate = 1e-6
    global_step = tf.Variable(0, trainable=False)
    decay_steps = 500
    max_gradient_norm = 5.0
    max_decode_step = None

    batch_size = 128  # 每批训练大小
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

    def build_single_cell(self):
        """构建一个单独的rnn cell
        Args:
            n_hidden: 隐藏层神经元数量
            use_residual: 是否使用residual wrapper
        """

        if self.config.cell_type == 'gru':
            cell_type = GRUCell
        else:
            cell_type = LSTMCell

        cell = cell_type(self.config.hidden_dim)

        if self.config.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob
            )

        if self.config.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def build_rnn_cell(self):
        """构建一个单独的编码器cell
        """
        return MultiRNNCell([
            self.build_single_cell()
            for _ in range(self.config.num_layers)
        ])  # 根据self.depth值生成RNN多层的结构网络

    # def build_rnn_model(self):
    #     """构建编码器
    #     """
    #     # print("多层rnn模型")
    #     embedding_inputs = self.input_embedding()
    #     print("embedding_inputs", embedding_inputs.shape)
    #     rnn_cell = self.build_rnn_cell()
    #     # 目前這部的具体用意还不知为什么，有待考察
    #     if self.config.use_residual:
    #         embedding_inputs = layers.dense(embedding_inputs,
    #                                         self.config.hidden_dim,
    #                                         activation=tf.nn.relu,
    #                                         use_bias=False,
    #                                         name='residual_projection'
    #                                         )
    #     inputs = embedding_inputs
    #
    #     if self.config.time_major:
    #         inputs = tf.transpose(inputs, (1, 0, 2))
    #
    #     if not self.config.bidirectional:
    #         # 单向 RNN
    #         (encoder_outputs, encoder_state) = tf.nn.dynamic_rnn(cell=rnn_cell,
    #                                                              inputs=inputs,
    #                                                              dtype=tf.float32,
    #                                                              time_major=self.config.time_major
    #                                                              )
    #     else:
    #         # 双向 RNN
    #         rnn_cell_bw = self.build_rnn_cell()
    #         ((encoder_fw_outputs, encoder_bw_outputs),
    #          (encoder_fw_state, encoder_bw_state)) = \
    #             tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell,
    #                                             cell_bw=rnn_cell_bw,
    #                                             inputs=inputs,
    #                                             dtype=tf.float32,
    #                                             time_major=self.config.time_major
    #                                             )
    #         # 首先合并两个方向 RNN 的输出
    #         encoder_outputs = tf.concat(
    #             (encoder_fw_outputs, encoder_bw_outputs), 2)
    #
    #         encoder_state = []
    #         for i in range(self.config.num_layers):
    #             encoder_state.append(encoder_fw_state[i])
    #             encoder_state.append(encoder_bw_state[i])
    #         encoder_state = tuple(encoder_state)
    #     return encoder_outputs, encoder_state



    def rnn(self):
        """CNN模型"""
        with tf.name_scope("rnn"):
            """构建编码器
                    """
            # print("多层rnn模型")
            embedding_inputs = self.input_embedding()
            print("embedding_inputs", embedding_inputs.shape)
            rnn_cell = self.build_rnn_cell()
            # 目前這部的具体用意还不知为什么，有待考察
            if self.config.use_residual:
                embedding_inputs = layers.dense(embedding_inputs,
                                                self.config.hidden_dim,
                                                activation=tf.nn.relu,
                                                use_bias=False,
                                                name='residual_projection'
                                                )
            inputs = embedding_inputs

            if self.config.time_major:
                inputs = tf.transpose(inputs, (1, 0, 2))

            if not self.config.bidirectional:
                # 单向 RNN
                (encoder_outputs, encoder_state) = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                                     inputs=inputs,
                                                                     dtype=tf.float32,
                                                                     time_major=self.config.time_major
                                                                     )
            else:
                # 双向 RNN
                rnn_cell_bw = self.build_rnn_cell()
                ((encoder_fw_outputs, encoder_bw_outputs),
                 (encoder_fw_state, encoder_bw_state)) = \
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell,
                                                    cell_bw=rnn_cell_bw,
                                                    inputs=inputs,
                                                    dtype=tf.float32,
                                                    time_major=self.config.time_major
                                                    )
                # 首先合并两个方向 RNN 的输出
                encoder_outputs = tf.concat(
                    (encoder_fw_outputs, encoder_bw_outputs), 2)

                encoder_state = []
                for i in range(self.config.num_layers):
                    encoder_state.append(encoder_fw_state[i])
                    encoder_state.append(encoder_bw_state[i])
                encoder_state = tuple(encoder_state)

            # _outputs, _ = self.build_rnn_model()
            print("_outputs", encoder_outputs.shape)
            last = encoder_outputs[:, -1, :]
            print("last", last.shape)

        with tf.name_scope("score"):
            cur_layer = last
            layer_dimension = [250, 100]  # 神经网络层节点的个数
            n_layers = len(layer_dimension)  # 神经网络的层数
            for i in range(0, n_layers):
                out_dimension = layer_dimension[i]
                fc = tf.layers.dense(inputs=cur_layer, \
                                     units=out_dimension, \
                                     activation=tf.nn.relu, \
                                     activity_regularizer=tf.contrib.layers.l2_regularizer(0.001), \
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), \
                                     # name='fc1'
                                     )
                cur_layer = tf.contrib.layers.dropout(fc, self.keep_prob)

            # 分类器，输出层
            self.logits = tf.layers.dense(cur_layer, self.config.num_classes, name='fc2')
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
