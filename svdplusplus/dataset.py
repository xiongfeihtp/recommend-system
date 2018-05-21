'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: Dataset.py
@time: 2018/5/21 下午7:12
@desc: shanghaijiaotong university
'''
import itertools
from scipy.sparse import csr_matrix
import numpy as np
from matrix import Matrix


class Dataset:
    def __init__(self, filename, k_fold=7, shuffle=True):
        self.N_all = 0
        self.filename = filename
        self.k_fold = k_fold
        self.shuffle = shuffle

    def read_rating(self):
        with open(self.filename) as f:
            raw_ratings = [self.parse(line) for line in itertools.islice(f, 0, None)]
        self.N_all = len(raw_ratings)
        return raw_ratings

    def parse(self, line):
        uid, iid, r, timestamp = line.split('\t')
        return uid, iid, float(r), timestamp

    def cv(self):
        raw_ratings = self.read_rating()
        if self.shuffle:
            np.random.shuffle(raw_ratings)
        stop = 0
        raw_len = len(raw_ratings)
        gap = raw_len // self.k_fold
        left = raw_len % self.k_fold
        # 23 5
        # 0-5 5-10 10-15 15-19 19-23
        for i in range(self.k_fold):
            print("current fold: {}".format(i + 1))
            start = stop
            stop += gap
            if i < left:
                stop += 1
            yield self.mapping(raw_ratings[:start] + raw_ratings[stop:]), raw_ratings[start:stop]

    def mapping(self, raw_train_ratings):
        uid_dict = {}
        iid_dict = {}
        user_index = 0
        item_index = 0
        row = []
        col = []
        data = []
        for uid, iid, r, _ in raw_train_ratings:
            if uid not in uid_dict:
                uid_dict[uid] = user_index
                user_index += 1
            if iid not in iid_dict:
                iid_dict[iid] = item_index
                item_index += 1
            row.append(uid_dict[uid])
            col.append(iid_dict[iid])
            data.append(r)
        sparse_matrix = csr_matrix((data, (row, col)))
        # 为稀疏矩阵定义一系列操作，方面编程
        return Matrix(sparse_matrix, uid_dict, iid_dict)

    def eval(self, algorithm):
        eval_results = []
        for i, (train_data, test_data) in enumerate(self.cv()):
            print("cv: {}".format(i))
            algorithm.train(train_data, self.N_all - len(test_data))
            eval_results.append(algorithm.estimate(test_data))
        return eval_results


if __name__ == "__main__":
    """
    test
    """
    dataset = Dataset('./data/u.data', k_fold=10)
    for train_data, test_data in dataset.cv():
        pass
