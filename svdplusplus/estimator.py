'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: estimator.py
@time: 2018/5/21 下午7:13
@desc: shanghaijiaotong university
'''
from __future__ import division, print_function
import numpy as np
from util import Timer
from tqdm import tqdm


class Estimator:
    def __init__(self):
        pass

    def train(self, train_dataset, all):
        self.train_dataset = train_dataset
        with Timer() as t:
            self._train()
        print("{} algorithm train process cost {} sec".format(self.__class__.__name__, t.interval))

    def _train(self):
        raise NotImplementedError

    def predict(self, u, i):
        raise NotImplementedError

    def estimate(self, raw_test_dataset):
        with Timer() as t:
            error = self._estimate(raw_test_dataset)

    def _estimate(self, raw_test_dataset):
        print("estimate...")
        users_mean = self.train_dataset.get_user_means()
        items_mean = self.train_dataset.get_item_means()
        all = len(raw_test_dataset)
        errors = []
        cur = 0
        valid_count = 0
        for raw_u, raw_i, r, _ in tqdm(raw_test_dataset):
            cur += 1
            has_raw_u = raw_u in self.train_dataset.uid_dict
            has_raw_i = raw_i in self.train_dataset.iid_dict
            if not has_raw_u and not has_raw_i:
                real, est = r, self.train_dataset.global_mean
            elif not has_raw_u:
                i = self.train_dataset.iid_dict[raw_i]
                real, est = r, items_mean[i]
            elif not has_raw_i:
                u = self.train_dataset.uid_dict[raw_u]
                real, est = r, users_mean[u]
            else:
                u = self.train_dataset.uid_dict[raw_u]
                i = self.train_dataset.iid_dict[raw_i]
                real, est = r, self.predict(u, i)
                valid_count += 1
            est = min(5, est)
            est = max(1, est)
            errors.append(real - est)
            self.progress(cur, all)
        return errors

    @staticmethod
    def progress(cur, all, bin=50):
        if cur % bin == 0 or cur == all:
            progress = 100 * (cur / all)
            print("progress: {:.2f}%".format(progress))
