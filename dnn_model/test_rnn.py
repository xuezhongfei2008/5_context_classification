#encoding:utf-8
import numpy as np

'''生成数据
就是按照文章中提到的规则，这里生成1000000个
'''
def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    '''根据规则生成Y'''
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -=0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)



if __name__=='__main__':
    X,Y=gen_data(10)
    print(X,"----",Y)