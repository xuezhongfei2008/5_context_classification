#encoding:utf-8
import random
import os

base_dir = '/opt/gongxf/python3_pj/Robot/4_cnn_simi/data'
train_dir = os.path.join(base_dir, 'train1.txt')
test_dir = os.path.join(base_dir, 'test1.txt')



with open('/opt/gongxf/python3_pj/Robot/4_cnn_simi/data/train.txt', 'r') as f:
    lines = f.readlines()

with open(train_dir, 'w') as fa, open(test_dir, 'w') as fb:
    for _ in range(4200):
        fa.write(lines.pop(random.randint(0, len(lines) - 1)))
    fb.writelines(lines)