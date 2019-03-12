# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle
base_path='/opt/algor/gongxf/python3_pj/nlp_practice/5_context_classification/dnn_model'

def load_embedding():
    #读取word2vec词向量
    embedding_array = np.load(base_path+"/data/embedding_word2vec.npy")
    #获取词汇大小
    vocab_size=embedding_array.shape[0]
    #回去词向量 shape
    embedding_dim=embedding_array.shape[1]
    #返回labels的种类数
    pickle_file = open(base_path+'/data/categories.pkl', 'rb')
    categories = pickle.load(pickle_file)
    categories_len=len(categories)
    return embedding_array,vocab_size,embedding_dim,categories_len

class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = load_embedding()[2]      # 词向量维度,使用gensim 设置的值
    seq_length = 8        # 序列长度
    num_classes = load_embedding()[3]         # 类别数
    # num_filters = 32        # 卷积核数目
    # kernel_size = 3         # 卷积核尺寸
    vocab_size = load_embedding()[1]       # 词汇表达小 使用gensim 设置的值

    # hidden_dim = 180        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10        # 总迭代轮次

    print_per_batch = 10    # 每多少轮输出一次结果
    save_per_batch = 100      # 每多少轮存入tensorboard

class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None,self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()
    def input_embedding(self):
        #词向量的转换# 词向量映射
        with tf.device('/cpu:0'):
            # 当trainable 值设为True时，该模型就是non_static,当trainable值设为False时模型就是static的
            embedding = tf.get_variable('embedding', trainable=True,initializer=load_embedding()[0])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_inputs=tf.expand_dims(embedding_inputs,-1)
        return  embedding_inputs

    def cnn(self):
        """CNN模型"""
        embedding_inputs=self.input_embedding()
        print("embedding_inputs",embedding_inputs.shape)
        pooled_outputs=[]
        filter_sizes = [[1, 300]]
        num_filters_total=0
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("cnn%s" % filter_size[0]):
                # filter_shape=[filter_size[0],self.config.embedding_dim,1,self.config.num_filters]
                filter_shape=[filter_size[0],self.config.embedding_dim,1,filter_size[1]]
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                # print("W",W.shape)
                # print("embedding_inputs",embedding_inputs.shape)
                conv=tf.nn.conv2d(
                    embedding_inputs,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv" )
                # print("conv",conv.shape)
                pooled=tf.nn.max_pool(
                    conv,
                    ksize=[1,self.config.seq_length-filter_size[0]+1,1,1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool")
                # print("pooled",pooled.shape)
                pooled_outputs.append(pooled)
            num_filters_total+=filter_size[1]
        self.h_pool=tf.concat(pooled_outputs,-1)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])    #映射成一个 num_filters_total 维的特征向量
        # print("self.h_pool_flat", self.h_pool_flat.shape)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # 隐藏层函数定义
            cur_layer = self.h_pool_flat
            layer_dimension = [500,250,125]  # 神经网络层节点的个数
            n_layers = len(layer_dimension)  # 神经网络的层数
            for i in range(0, n_layers):
                out_dimension = layer_dimension[i]
                fc = tf.layers.dense(inputs=cur_layer,\
                                     units=out_dimension, \
                                     activation=tf.nn.relu\
                                     # activity_regularizer=tf.contrib.layers.l2_regularizer(0.001), \
                                     # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),\
                                     # name='fc1'
                                     )
                cur_layer = tf.contrib.layers.dropout(fc,self.keep_prob)

            # 输出层,分类器
            self.logits = tf.layers.dense(cur_layer, self.config.num_classes, name='fc2')
            self.logits_softmax=tf.nn.softmax(self.logits)
            # self.logits1 = tf.nn.local_response_normalization(self.logits,dim = 0)
            # print("self.logits", self.logits.shape)
            self.y_pred = tf.argmax(self.logits_softmax, 1)  # 预测类别
            # print("self.y_pred",self.y_pred.shape)

        with tf.name_scope("loss"):
            # 使用优化方式，损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
        with tf.name_scope("optimize"):
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            correct_pred=tf.equal(self.y_pred,tf.argmax(self.input_y,1))
            self.acc=tf.reduce_mean(tf.cast(correct_pred,"float"),name="accuracy")




