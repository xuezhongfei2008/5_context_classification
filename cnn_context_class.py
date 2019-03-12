from context_queue import robot_context_queue
from jieba_cut import *
import sys
sys.path.append('/opt/gongxf/python3_pj/nlp_practice/5_context_classification/dnn_model/cnn_conv2d')
from dnn_model.cnn_conv2d.cnn_predict import CNN_Predict
import re
from keyword_nv import robot_keyword_nv


class CNN_Context(object):
    """
         功能：智能客服语义分析引擎-fasttext上下文分类
         输入：
         输出：
         方法：构建存储上下文的队列
    """

    def __init__(self,
                 threshold=0.8  # 分类阈值
                 ):
        ###   STEP1：定义初始化数据  ###
        self.threshold = threshold
        self.cnn_model=CNN_Predict()

    def outstr_str(self,text):
        outstr = ''
        for word in jieba_cut(text):
            if word != '\t':
                outstr += word
                # outstr += " "
        return outstr

    def cnn_single(self,text):
        outstr = self.outstr_str(text)
        cate,prob = self.cnn_model.predict(outstr)
        return cate

    def cnn_context(self,Text, userId, requesId):
        robot_context_queue.context_queue(Text, userId, requesId)
        print("robot_context_queue.UserText", robot_context_queue.UserText)
        len_userId_UserText = len(robot_context_queue.UserText[userId])
        if len_userId_UserText == 1:
            text_0 = robot_context_queue.UserText[userId][len_userId_UserText - 1][0]  # 当前输入值
            outstr_0 = self.outstr_str(text_0)
            cate_0,prob_0 = self.cnn_model.predict(outstr_0)
            # print("outstr_0", outstr_0)
            # print("cate_0", cate_0)
            return cate_0
        elif len_userId_UserText == 2:
            text_0 = robot_context_queue.UserText[userId][len_userId_UserText - 1][0]  # 当前输入值
            text_1 = robot_context_queue.UserText[userId][len_userId_UserText - 2][0]  # 前一条数据
            outstr_0 = self.outstr_str(text_0)
            cate_0,prob_0 = self.cnn_model.predict(outstr_0)
            # print("outstr_0", outstr_0)
            # print("cate_0", cate_0)
            if prob_0 >= self.threshold:
                return cate_0
            else:
                text_01 = text_1 + text_0
                outstr_01 = self.outstr_str(text_01)
                cate_01,prob_01 = self.cnn_model.predict(outstr_01)
                # print("outstr_01", outstr_01)
                # print("cate_01", cate_01)
                return cate_01
        elif len_userId_UserText == 3:
            text_0 = robot_context_queue.UserText[userId][len_userId_UserText - 1][0]  # 当前输入值
            text_1 = robot_context_queue.UserText[userId][len_userId_UserText - 2][0]  # 前一条数据
            text_2 = robot_context_queue.UserText[userId][len_userId_UserText - 3][0]  # 前前一条数据
            outstr_0 = self.outstr_str(text_0)
            cate_0,prob_0 = self.cnn_model.predict(outstr_0)
            # print("outstr_0", outstr_0)
            # print("cate_0", cate_0)
            if prob_0 >= self.threshold:
                return cate_0
            else:
                text_01 = text_1 + text_0
                outstr_01 = self.outstr_str(text_01)
                cate_01,prob_01 = self.cnn_model.predict(outstr_01)
                # print("outstr_01", outstr_01)
                # print("cate_01", cate_01)
                if prob_01 >= self.threshold:
                    return cate_01
                else:
                    text_012 = text_2 + text_1 + text_0
                    outstr_012 = self.outstr_str(text_012)
                    cate_012,prob_012 = self.cnn_model.predict(outstr_012)
                    # print("outstr_012", outstr_012)
                    # print("cate_012", cate_012)
                    return cate_012
        else:
            cate,prob = self.cnn_single(Text)
            return cate

    def cnn_keyword_single(self,text):
        outstr=robot_keyword_nv.keyword_nv_cnn(text)
        # outstr = self.outstr_str(keyword_str)
        cate,prob = self.cnn_model.predict(outstr)
        return cate

    def cnn_keyword_context(self,Text, userId, requesId):
        robot_context_queue.context_queue(Text, userId, requesId)
        # print("robot_context_queue.UserText", robot_context_queue.UserText)
        len_userId_UserText = len(robot_context_queue.UserText[userId])
        if len_userId_UserText == 1:
            text_0 = robot_context_queue.UserText[userId][len_userId_UserText - 1][0]  # 当前输入值
            outstr_0 = robot_keyword_nv.keyword_nv_cnn(text_0)
            # outstr_0 = self.outstr_str(text_0)
            cate_0,prob_0 = self.cnn_model.predict(outstr_0)
            # print("outstr_0", outstr_0)
            # print("cate_0", cate_0)
            return cate_0
        elif len_userId_UserText == 2:
            text_0 = robot_context_queue.UserText[userId][len_userId_UserText - 1][0]  # 当前输入值
            text_1 = robot_context_queue.UserText[userId][len_userId_UserText - 2][0]  # 前一条数据
            outstr_0 = robot_keyword_nv.keyword_nv_cnn(text_0)
            # outstr_0 = self.outstr_str(text_0)
            cate_0,prob_0 = self.cnn_model.predict(outstr_0)
            # print("outstr_0", outstr_0)
            # print("cate_0", cate_0)
            if prob_0 >= self.threshold:
                return cate_0
            else:
                text_01 = text_1 + text_0
                outstr_01 = robot_keyword_nv.keyword_nv_cnn(text_01)
                # outstr_01 = self.outstr_str(text_01)
                cate_01,prob_01 = self.cnn_model.predict(outstr_01)
                # print("outstr_01", outstr_01)
                # print("cate_01", cate_01)
                return cate_01
        elif len_userId_UserText == 3:
            text_0 = robot_context_queue.UserText[userId][len_userId_UserText - 1][0]  # 当前输入值
            text_1 = robot_context_queue.UserText[userId][len_userId_UserText - 2][0]  # 前一条数据
            text_2 = robot_context_queue.UserText[userId][len_userId_UserText - 3][0]  # 前前一条数据
            outstr_0 = robot_keyword_nv.keyword_nv_cnn(text_0)
            # outstr_0 = self.outstr_str(text_0)
            cate_0,prob_0 = self.cnn_model.predict(outstr_0)
            # print("outstr_0", outstr_0)
            # print("cate_0", cate_0)
            if prob_0 >= self.threshold:
                return cate_0
            else:
                text_01 = text_1 + text_0
                outstr_01 = robot_keyword_nv.keyword_nv_cnn(text_01)
                # outstr_01 = self.outstr_str(text_01)
                cate_01,prob_01 = self.cnn_model.predict(outstr_01)
                # print("outstr_01", outstr_01)
                # print("cate_01", cate_01)
                if prob_01 >= self.threshold:
                    return cate_01
                else:
                    text_012 = text_2 + text_1 + text_0
                    outstr_012 = robot_keyword_nv.keyword_nv_cnn(text_012)
                    # outstr_012 = self.outstr_str(text_012)
                    cate_012,prob_012 = self.cnn_model.predict(outstr_012)
                    # print("outstr_012", outstr_012)
                    # print("cate_012", cate_012)
                    return cate_012
        else:
            cate,prob = self.cnn_keyword_single(Text)
            return cate

if __name__=='__main__':
    robot_cnn_context=CNN_Context()
    while True:
        text=input("请输入测试问题：")
        userid=input("请输入用户编号：")
        requid=input("请输入session代码：")
        cate = robot_cnn_context.cnn_keyword_context(text, userid, requid)
        # cate = robot_cnn_context.fasttext_keyword_single(text)
        # cate = robot_cnn_context.cnn_keyword_single(text)
        print(cate)




