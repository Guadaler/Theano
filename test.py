#!/usr/bin/env python
# encoding: utf-8

# -------------------------------------------
# 功能：
# Author: zx
# Software: PyCharm Community Edition
# File: test.py
# Time: 16-12-30 上午11:42
# -------------------------------------------
from theano import shared
from theano import function
import theano.tensor as T
import numpy as np
import theano


def compute_tanh():
    rng = np.random

    # 定义自变量
    x, w = T.dmatrices('x', 'w')
    b = T.dvector('b')

    # 定义循环体，计算因变量
    y, u = theano.scan(lambda i, w, b: T.tanh(T.dot(i, w) + b), sequences=x, non_sequences=[w, b])

    # 构造完整计算方法
    result = function([x, w, b], y)

    # 初始化数据
    x_t = rng.rand(4, 5)
    w_t = rng.rand(5, 4)
    b_t = rng.rand(4)

    # 喂数据
    print x_t
    print w_t
    print result(x_t, w_t, b_t)


def compute_ak():
    rng = np.random
    k = T.iscalar('k')
    a = T.dvector('a')
    y, u = theano.scan(lambda i, j: i * j, non_sequences=a, outputs_info=T.ones_like(a), n_steps=k)

    result = function([a, k], y)

    x = rng.randint(10, size=5)
    res = result(x, 3)
    print x
    print res[-1]
    print res

#
# def compute_ak():
#     rng = np.random
#     k = T.iscalar('k')
#     a = T.dvector('a')
#     c = T.dvector('c')
#     y, u = theano.scan(lambda i, j: i * j, non_sequences=a, outputs_info=c, n_steps=k)
#
#     result = function([a, k, c], y)
#
#     x = rng.randint(10, size=5)
#     x2 = rng.randint(10, size=5)
#     res = result(x, 3,x2)
#     print "a ",x
#     print "c ",x2
#     print res[0]
#     print res[1]
#     print res[2]


# compute_tanh()
compute_ak()

state = shared(0)  # shared变量state
inc = T.iscalar('inc')  # 整形标量inc
state = state + inc

# acc = function([inc],state, updates=[(state, state+inc)])
acc = function([inc], state)

print acc
print type(acc)
print acc(5)
