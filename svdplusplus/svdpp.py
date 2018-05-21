'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: SVD++.py
@time: 2018/5/21 下午6:29
@desc: shanghaijiaotong university
'''
from __future__ import division, print_function
import itertools
import os
from scipy.sparse import csr_matrix
import numpy as np
from estimator import Estimator
from tqdm import tqdm


class SVDpp(Estimator):
    def __init__(self, n_factors=20, n_epochs=20, lr=0.007, reg=0.002):
        """
        :param n_factors: latent vector dim
        :param n_epochs: train epochs
        :param lr: learning rate
        :param reg: regularization rate
        """
        super(SVDpp, self).__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def train(self, train_dataset, all):
        user_num = train_dataset.matrix.shape[0]
        item_num = train_dataset.matrix.shape[1]
        self.global_mean = train_dataset.global_mean
        self.train_dataset = train_dataset

        self.bu = np.zeros(user_num, np.double)
        self.bi = np.zeros(item_num, np.double)

        self.p = np.random.randn(user_num, self.n_factors) * 0.05
        self.q = np.random.randn(item_num, self.n_factors) * 0.05

        self.y = np.random.randn(item_num, self.n_factors) * 0.05

        for current_epoch in range(self.n_epochs):
            print("current_epoch: {}".format(current_epoch))
            cur = 0
            for u, i, r in train_dataset.all_ratings() :
                u_item = train_dataset.get_user(u)[0]
                N_u_item = len(u_item)
                sqrt_N = np.sqrt(N_u_item)

                u_impl_pref = np.sum(self.y[u_item], axis=0) / sqrt_N

                rp = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + u_impl_pref)
                e_ui = r - rp
                self.bu[u] += self.lr * (e_ui - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e_ui - self.reg * self.bi[i])
                self.p[u] += self.lr * (e_ui * self.q[i] - self.reg * self.p[u])
                self.q[i] += self.lr * (e_ui * (self.p[u] + u_impl_pref) - self.reg * self.q[i])

                for j in u_item:
                    self.y[j] += self.lr * (e_ui * self.q[j] / N_u_item - self.reg * self.y[j])
                cur += 1
                self.progress(cur, all)

    def predict(self, u, i):
        u_item = self.train_dataset.get_user(u)[0]
        N_u_item = len(u_item)
        sqrt_N = np.sqrt(N_u_item)
        u_impl_pref = np.sum(self.y[u_item], axis=0) / sqrt_N
        est = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + u_impl_pref)
        return est
