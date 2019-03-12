import time
import numpy as np


class Context_Queue(object):
    """
         功能：智能客服语义分析引擎-上下文存储机制队列
         输入：
         输出：
         方法：构建存储上下文的队列
    """

    def __init__(self,
                 size=3,
                 time_interval=0.5  # 小时为单位，超过time_interval时间未操作清除该数据
                 ):
        ###   STEP1：定义初始化数据  ###
        self.size = size
        self.time_interval = time_interval
        self.UserText = {}
        self.UserText_Time = {}

    def is_requesId(self, requesId, userId_TextList):
        # 判断队列中是否是同一个session会话
        for text in userId_TextList:
            if requesId in text:
                return True
            else:
                return False

    def is_content(self, Text, userId_TextList):
        # 判断新传入得text是否与前两个队列数据相同
        if len(userId_TextList) >= 2:
            if Text in userId_TextList[len(userId_TextList) - 1] and Text in userId_TextList[len(userId_TextList) - 2]:
                return True
        else:
            return False

    # 删除超时的队列数据（time_interval单位为时间）
    def delete_usertext(self):
        if self.UserText:
            now_time = time.time()
            Time_np = np.array(list(self.UserText_Time.values()))
            Timeout_queue = Time_np[:, 1] < str(now_time - self.time_interval * 60 * 60)
            result_time = np.column_stack((Time_np, Timeout_queue))
            for ii in result_time:
                if ii[2] == str(True):
                    if ii[0] in self.UserText.keys():
                        self.UserText.pop(ii[0])
                        self.UserText_Time.pop(ii[0])
        # return UserText, UserText_Time

    def clear_all(self):
        self.UserText.clear()
        self.UserText_Time.clear()

    def context_queue(self, Text, userId, requesId):
        Text_session = (Text, requesId)
        Time_session = (userId, time.time())
        if userId in self.UserText.keys() and userId in self.UserText_Time.keys():
            if self.is_requesId(requesId, self.UserText[userId]):
                if self.is_content(Text, self.UserText[userId]):
                    print("重复三遍以上，转人工吧。。。")
                    # return "转人工"
            else:
                self.UserText[userId].clear()
        else:
            self.UserText[userId] = []
            self.UserText_Time[userId] = []
        self.UserText[userId].append(Text_session)
        self.UserText_Time[userId] = Time_session
        if self.UserText[userId] is not None and len(self.UserText[userId]) <= self.size:
            pass
        else:
            self.UserText[userId].remove(self.UserText[userId][0])
        # print("TextList", self.UserText[userId])
        # return UserText, UserText_Time


robot_context_queue = Context_Queue(size=3, time_interval=0.5)
