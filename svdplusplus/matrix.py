'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: Matrix.py
@time: 2018/5/21 下午7:11
@desc: shanghaijiaotong university
'''

import numpy as np


class Matrix:
    def __init__(self, sparse_matrix, uid_dict=None, iid_dict=None):
        self.matrix = sparse_matrix.tocsc()
        self._global_mean = None
        coo_matrix = sparse_matrix.tocoo()
        self.uids = set(coo_matrix.row)
        self.iids = set(coo_matrix.col)
        self.uid_dict = uid_dict
        self.iid_dict = iid_dict

    def all_ratings(self):
        coo_matrix = self.matrix.tocoo()
        return zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)

    def get_user(self, user_index):
        ratings = self.matrix.getrow(user_index).tocoo()
        return ratings.col, ratings.data

    def get_item(self, item_index):
        ratings = self.matrix.getcol(item_index).tocoo()
        return ratings.row, ratings.data

    def get_uids(self):
        return np.unique(self.matrix.tocoo().row)

    def get_user_means(self):
        users_mean = {}
        for u in self.get_uids():
            users_mean[u] = np.mean(self.get_user(u)[1])

    def get_iids(self):
        return np.unique(self.matrix.tocoo().col)

    def get_item_means(self):
        items_mean = {}
        for i in self.get_iids():
            items_mean[i] = np.mean(self.get_item(i)[1])

    @property
    def global_mean(self):
        self._global_mean = np.mean(self.matrix.data)
        return self._global_mean
