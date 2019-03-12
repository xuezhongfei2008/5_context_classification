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
    def __init__(self):
        self.embedding_dim = load_embedding()[2]  # 词向量维度,使用gensim 设置的值
        self.seq_length = 10  # 序列长度
        self.num_classes = load_embedding()[3]  # 类别数
        self.vocab_size = load_embedding()[1]  # 词汇表达小 使用gensim 设置的值

        # 模型构建参数
        # self.num_layers = 2  # RNN 层数
        # self.hidden_dim = 250   #RNN神经元个数

        self.hidden_dim_num =[300,300]  # RNN神经元个数

        self.layer_dimension=[500,250,125]  # 全连接层神经元,中间层

        self.cell_type = 'gru'  # lstm 或者 gru
        self.use_residual = False
        self.use_dropout = True
        self.time_major = False
        self.bidirectional = False
        self.dropout_keep_prob = 0.5  # dropout保留比例

        # 模型优化参数
        self.optimizer = 'adam'
        self.learning_rate = 1e-3
        self.min_learning_rate = 1e-6
        self.global_step = tf.Variable(0, trainable=False)
        self.decay_steps = 500
        self.max_gradient_norm = 5.0
        self.max_decode_step = None

        #训练参数
        self.batch_size = 128  # 每批训练大小
        self.num_epochs = 10  # 总迭代轮次

        #打印参数
        self.print_per_batch = 10  # 每多少轮输出一次结果
        self.save_per_batch = 10  # 每多少轮存入tensorboard


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
            embedding = tf.get_variable('embedding',trainable=True, initializer=load_embedding()[0])
            # embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],initializer=load_embedding()[0])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return embedding_inputs

    # def build_single_cell(self):
    #     """构建一个单独的rnn cell
    #     Args:
    #         n_hidden: 隐藏层神经元数量
    #         use_residual: 是否使用residual wrapper
    #     """
    #
    #     if self.config.cell_type == 'gru':
    #         cell_type = GRUCell
    #     else:
    #         cell_type = LSTMCell
    #
    #     cell = cell_type(self.config.hidden_dim)
    #
    #     if self.config.use_dropout:
    #         cell = DropoutWrapper(
    #             cell,
    #             dtype=tf.float32,
    #             output_keep_prob=self.keep_prob
    #         )
    #
    #     if self.config.use_residual:
    #         cell = ResidualWrapper(cell)
    #
    #     return cell

    def build_single_cell_num(self,rnn_num):
        """构建一个单独的rnn cell
        Args:
            n_hidden: 隐藏层神经元数量
            use_residual: 是否使用residual wrapper
        """

        if self.config.cell_type == 'gru':
            cell_type = GRUCell
        else:
            cell_type = LSTMCell

        cell = cell_type(rnn_num)

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
        cells= MultiRNNCell([
            self.build_single_cell_num(num)
            for num in self.config.hidden_dim_num
        ])  # 根据self.depth值生成RNN多层的结构网络
        return cells
        #     MultiRNNCell([
        #     self.build_single_cell()
        #     for _ in range(self.config.num_layers)
        # ])  # 根据self.depth值生成RNN多层的结构网络

    def build_rnn_model(self):
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
        # inputs = embedding_inputs
        inputs=self.batch_normalization(embedding_inputs)

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
        return encoder_outputs, encoder_state

    def init_optimizer(self):
        """初始化优化器
        支持的方法有 sgd, adadelta, adam, rmsprop, momentum
        """

        # 学习率下降算法
        learning_rate = tf.train.polynomial_decay(
            self.config.learning_rate,
            self.config.global_step,
            self.config.decay_steps,
            self.config.min_learning_rate,
            power=0.5
        )
        self.current_learning_rate = learning_rate

        # 设置优化器,合法的优化器如下
        # 'adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'
        trainable_params = tf.trainable_variables()
        if self.config.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate)
        elif self.config.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        elif self.config.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate)
        elif self.config.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif self.config.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)

        # # Compute gradients of loss w.r.t. all trainable variables
        # gradients = tf.gradients(self.loss, trainable_params)
        # # Clip gradients by a given maximum_gradient_norm
        # clip_gradients, _ = tf.clip_by_global_norm(
        #     gradients, self.config.max_gradient_norm)
        # # Update the model
        # optim = self.opt.apply_gradients(
        #     zip(clip_gradients, trainable_params),
        #     global_step=self.config.global_step)
        optim = self.opt.minimize(self.loss)

        return optim

    def calculation_loss(self):
        """初始化优化器
        支持的方法有 sgd, adadelta, adam, rmsprop, momentum
        """
        cur_layer = self.last
        cur_layer=self.batch_normalization(cur_layer)
        # layer_dimension = [250,100]  # 神经网络层节点的个数
        n_layers = len(self.config.layer_dimension)  # 神经网络的层数
        for i in range(0, n_layers):
            out_dimension = self.config.layer_dimension[i]
            fc = tf.layers.dense(inputs=cur_layer, \
                                 units=out_dimension, \
                                 activation=tf.nn.relu, \
                                 activity_regularizer=tf.contrib.layers.l2_regularizer(0.001), \
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), \
                                 # name='fc1'
                                 )
            cur_layer = tf.contrib.layers.dropout(fc, self.keep_prob)
            cur_layer = self.batch_normalization(cur_layer)

        # 分类器，输出层
        logits = tf.layers.dense(cur_layer, self.config.num_classes, name='fc2')
        y_pred = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别
        # 使用优化方式，损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
        loss = tf.reduce_mean(cross_entropy)

        return logits, y_pred, loss

    def batch_normalization(self,img):
        axis = list(range(len(img.shape) - 1))
        wb_mean, wb_var = tf.nn.moments(img, axis)
        scale = tf.Variable(tf.ones([1]))
        offset = tf.Variable(tf.zeros([1]))
        variance_epsilon = 0.001
        Wx_plus_b = tf.nn.batch_normalization(img, wb_mean, wb_var, offset, scale, variance_epsilon)
        return Wx_plus_b


    # def cnn_conv_pool(self,inputs):
    #     embedding_inputs = tf.expand_dims(inputs, -1)
    #     pooled_outputs = []
    #     filter_sizes = [[1, 128], [2, 128], [3, 128], [4, 128]]
    #     num_filters_total = 0
    #     for i, filter_size in enumerate(filter_sizes):
    #         with tf.name_scope("cnn%s" % filter_size[0]):
    #             filter_shape = [filter_size[0], self.config.embedding_dim, 1, filter_size[1]]
    #             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
    #             conv = tf.nn.conv2d(
    #                 embedding_inputs,
    #                 W,
    #                 strides=[1, 1, 1, 1],
    #                 padding="VALID",
    #                 name="conv")
    #             pooled = tf.nn.max_pool(
    #                 conv,
    #                 ksize=[1, self.config.seq_length - filter_size[0] + 1, 1, 1],
    #                 strides=[1, 1, 1, 1],
    #                 padding="VALID",
    #                 name="pool")
    #             pooled_outputs.append(pooled)
    #         num_filters_total += filter_size[1]
    #     self.h_pool = tf.concat(pooled_outputs, -1)
    #     self.last = tf.reshape(self.h_pool, [-1, num_filters_total])  # 映射成一个 num_filters_total 维的特征向量
    #
    #     return self.last

    def rnn(self):
        """CNN模型"""
        _outputs, _ = self.build_rnn_model()
        embedding_inputs = tf.expand_dims(_outputs, -1)
        print("embedding_inputs",embedding_inputs.shape)
        pooled_outputs = []
        filter_sizes = [[1, 128], [2, 128], [3, 128], [4, 128]]
        num_filters_total = 0
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cnn%s" % filter_size[0]):
                filter_shape = [filter_size[0], self.config.embedding_dim, 1, filter_size[1]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                conv = tf.nn.conv2d(
                    embedding_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                pooled = tf.nn.max_pool(
                    conv,
                    ksize=[1, self.config.seq_length - filter_size[0] + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)
            num_filters_total += filter_size[1]
        self.h_pool = tf.concat(pooled_outputs, -1)
        self.last = tf.reshape(self.h_pool, [-1, num_filters_total])  # 映射成一个 num_filters_total 维的特征向量

        print("last", self.last.shape)
        # 计算损失函数
        self.logits, self.y_pred, self.loss = self.calculation_loss()
        # 优化
        self.optim = self.init_optimizer()
        # 准确率
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
