# -*- coding: utf-8 -*-
from gensim.models import Word2Vec, FastText
import jieba
import pickle
import numpy as np
import pandas as pd

base_path='/opt/gongxf/python3_pj/nlp_practice/5_context_classification/dnn_model'
jieba.load_userdict("/opt/gongxf/python3_pj/Robot/original_data/finWordDict.txt")


# 加载训练好的词向量、建立词汇文件
def bulid_embedding():
    # model = FastText.load('/opt/gongxf/python3_pj/Robot/2_fasttext2vec/all_session/fasttext_cbow.model')
    model=Word2Vec.load('/opt/gongxf/python3_pj/Robot/1_word2vec/all_session0704/cbow_dia.model')
    # print(model.vocabulary)
    file_vocab = open(base_path+"/data/vocab_word2vec.txt", 'w', encoding='utf-8')
    embedding_list = []
    for j in model.wv.vocab.keys():
        # 保存模型里面的词汇
        file_vocab.write(j + '\n')
        embedding_list.append(model[j])
    embedding = np.array(embedding_list)
    # 保存词向量文件
    np.save(base_path+"/data/embedding_word2vec.npy", embedding)
    file_vocab.close()


# 生产训练数据 并保存labels
def build_traindata():
    # 添加机器人知识库数据

    # 生产训练数据 labels：标准问题、labels：相似问题
    df=pd.read_table(base_path+"/data/train_keyword.txt",names=['calss','question'],sep='\t')
    print("df",df.head(1))
    categories=[]
    for ii in range(len(df)):
        categories.append(df["calss"][ii])
    categories = list(set(categories))
    print("categories",categories)
    # 保存categories：labels列表
    pickle_file = open(base_path+'/data/categories.pkl', 'wb')
    pickle.dump(categories, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    # bulid_embedding()
    build_traindata()
