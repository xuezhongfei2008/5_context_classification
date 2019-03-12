# encoding:utf-8
"""
__title__ = '__init__.py'
__author__ = 'Gongxuefei'
__mtime__ = '2018/7/23'
"""
import re
import jieba.posseg as pseg
from jieba_cut import *


class Keyword_NV(object):
    """
         功能：智能客服语义分析引擎-提前关键字
         输入：
         输出：
    """

    def __init__(self, **kwargs):
        ###  STEP1：公用数据部分 ###

        ##加载问题类别：标准问题{问题分类：[问题1，问题2，问题3]}
        self.stopwords = stopwords

    ###  STEP3：功能模块函数  ###
    def remove_rr(self,doc):
        pattern = ('^.*?[我你他她它您亲].*?')
        return re.compile(pattern).findall(doc)

    def keyword_nv(self, text):
        """
            功能：【关键词提取】
            输入：原始问题
            输出：关键词
        """
        # setp3: 提取nv关键词，去除人称代词杂质
        corpus0 = []
        s_cut = list(pseg.cut(text))
        postag = ['n', 'nr', 'nt', 'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi', 'vl', 'vg', 'x']
        for ii in s_cut:
            if (ii.flag in postag) and (ii.word not in self.stopwords) and (not self.remove_rr(ii.word)):
                corpus0.append(ii.word)
        keyword_str =''
        if corpus0:
            for text1 in corpus0:
                keyword_str += text1
                keyword_str +=' '
        else:
            keyword_str=text
        return keyword_str

    def keyword_nv_cnn(self, text):
        """
            功能：【关键词提取】
            输入：原始问题
            输出：关键词
        """
        # setp3: 提取nv关键词，去除人称代词杂质
        corpus0 = []
        s_cut = list(pseg.cut(text))
        postag = ['n', 'nr', 'nt', 'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi', 'vl', 'vg', 'x']
        for ii in s_cut:
            if (ii.flag in postag) and (ii.word not in self.stopwords) and (not self.remove_rr(ii.word)):
                corpus0.append(ii.word)
        keyword_str =''
        if corpus0:
            for text1 in corpus0:
                keyword_str += text1
        else:
            keyword_str=text
        return keyword_str

# 实例化语义分析模型
robot_keyword_nv = Keyword_NV()

