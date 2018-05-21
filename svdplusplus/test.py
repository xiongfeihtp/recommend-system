'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: test.py
@time: 2018/5/21 下午8:43
@desc: shanghaijiaotong university
'''
from dataset import Dataset
from svdpp import SVDpp

if __name__ == "__main__":
    dataset = Dataset('./data/u.data')
    dataset.eval(SVDpp())