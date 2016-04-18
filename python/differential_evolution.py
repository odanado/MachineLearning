#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pylab
from numpy.random import *

# sin波を多項式の和で近似する
class PolynomialApproximate:
    # n: 学習データの数
    def __init__(self, n):
        self.x = np.arange(0.0, 1 + 1.0 / (n - 1), 1.0 / (n - 1))
        self.y = np.sin(2.0 * np.pi * self.x)

        """
        # ノイズを乗せる
        for i in range(len(self.y)):
            self.y[i] += normal(0, 0.5)
        """
        self.CR = 0.5
        self.scaling = 0.6

        seed(0)

    # 多項式の和を取る
    def eval(self, x, w):
        idx = np.rot90(np.tile(np.arange(len(w)), (len(x), 1)), 3)
        return np.dot(w, np.power(x, idx))

    # 最小化したい関数
    def func(self, w):
        return np.sum(np.power(self.y - self.eval(self.x, w), 2))
    
    def error(self, w):
        return np.sum(np.power(np.sin(2.0 * np.pi * self.x) - self.eval(self.x, w), 2))


    
    #[0, n) でiと異なる3つの整数をランダムに生成する
    def select_randomly(self, i, n):
        pos = []
        while len(pos) < 3:
            p = randint(n)
            if p not in pos and i != p:
                pos.append(p)

        return pos

    def is_uniform(self, ws):
        res = True
        for i in range(len(ws)):
            if abs(self.func(ws[0]) - self.func(ws[i])) > 1e-6:
                res = False
                break
        return res

    # 反復回数, 係数の数, 解集団の数
    def run(self, iteration, n, m):
        # [-1, 1]の一様乱数を生成して解集団を初期化
        ws = rand(m, n) * 10 - 5

        for t in range(iteration):
            for i in range(len(ws)):
                pos = self.select_randomly(i, len(ws))
                j = randint(0, len(ws[0]))
                v = ws[pos[0]] + self.scaling * (ws[pos[1]] - ws[pos[2]])
                new_w = np.zeros(len(ws[0]))

                for k in range(len(ws[0])):
                    if k == 0 or rand() < self.CR:
                        new_w[j] = v[j]
                    else:
                        new_w[j] = ws[i][j]
                    j = (j + 1) % len(ws[0])

                if self.func(new_w) < self.func(ws[i]):
                    ws[i] = new_w

        return ws

if __name__ == '__main__':
    pa = PolynomialApproximate(20)
    ws = pa.run(10000, 9, 30)
    pylab.plot(pa.x, np.sin(2.0 * np.pi * pa.x))
    w = ws[0]
    for i in range(len(ws)):
        if pa.func(ws[i]) < pa.func(w):
            w = ws[i]
    pylab.plot(pa.x, pa.eval(pa.x, w))
    print(pa.error(w))
    # pylab.show()

