'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: util.py
@time: 2018/5/21 下午7:15
@desc: shanghaijiaotong university
'''
import time


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.clock()
        self.interval = self.end - self.start
