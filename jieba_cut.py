import jieba
jieba.load_userdict('/opt/gongxf/python3_pj/nlp_practice/5_context_classification/sn_ori_data/finWordDict.txt')

stopwords = [line.strip() for line in open('/opt/gongxf/python3_pj/nlp_practice/5_context_classification/sn_ori_data/stop_words.txt', 'r', encoding='utf-8').readlines()]
stopwords.append(' ')
stopwords.append(',')
stopwords.append('\n')
stopwords.append('\\n')


def jieba_cut(text):
    """
        功能：调用结巴分词函数,去停用词
        输入：标准问题
        输出：问题分词后的List
    """
    text_words = jieba.cut(text, cut_all=False)
    text_list = [word for word in text_words if word not in stopwords]
    return text_list

