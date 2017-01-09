#!/usr/bin/env python
# encoding: utf-8

# -------------------------------------------
# 功能：
# Author: zx
# Software: PyCharm Community Edition
# File: text_process.py
# Time: 16-12-29 上午11:02
# -------------------------------------------
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano
from gensim.models import Word2Vec

wemb_0 = []
word_index_0 = {}


# dict = {}

def init_embedding():
    global wemb_0
    global word_index_0

    word_index = {}

    # 初始化 wemb
    w2vmodel_path = "/home/zhangxin/work/workplace_python/theano/main/sentiment/word2vec_model/w2cmodel_lcm"
    w2vmodel = Word2Vec.load(w2vmodel_path)
    vocab = w2vmodel.vocab
    count = 0
    temp = []
    for v in vocab:
        count += 1
        vector = w2vmodel[v]
        temp.append(vector)
        word_index[v] = count

    wemb = numpy.array(temp)

    wemb_0 = wemb
    word_index_0 = word_index

    return wemb, word_index


def get_data_2():
    init_embedding()
    data_pos = []
    data_neg = []

    data_pos_path = '/home/zhangxin/文档/市场情绪分析/文献/依存句法/兰秋军/TotalCorpus/pos/seg_positive.txt'
    data_neg_path = '/home/zhangxin/文档/市场情绪分析/文献/依存句法/兰秋军/TotalCorpus/neg/seg_negative.txt'

    with open(data_pos_path) as f:
        flist = f.readlines()
        for d in flist:
            d = d.decode("gbk").strip("\n").replace(u"，", "").split(" ")
            data_pos.append(d)

    with open(data_neg_path) as f:
        flist = f.readlines()
        for d in flist:
            d = d.decode("gbk").strip("\n").replace(u"，", "").split(" ")
            data_neg.append(d)

    train = []
    for d in data_pos:
        temp = []
        for word in d:
            if word in word_index_0:
                index = word_index_0.get(word)
                temp.append(index)

        if len(temp) > 0:
            train.append((1, temp))

    for d in data_neg:
        temp = []
        # print " ".join(d)
        for word in d:
            if word in word_index_0:
                index = word_index_0.get(word)
                temp.append(index)

        # if 1 in temp:
        #     temp.remove(1)
        if len(temp) > 0:
            train.append((0, temp))

    index_word = dict({(word_index_0[k], k) for k in word_index_0.keys()})

    for xx in train:
        print xx
        # print " ".join([index_word[xxx] for xxx in xx])

    return train


def get_data_test_2():
    get_data()
    data_pos = []

    data_pos_path = '/home/zhangxin/文档/市场情绪分析/文献/依存句法/兰秋军/TotalCorpus_test/test.txt'

    with open(data_pos_path) as f:
        flist = f.readlines()
        for d in flist:
            d = d.decode("utf-8").strip("\n").replace(u"，", "").split(" ")
            data_pos.append(d)

    train = []
    for d in data_pos:
        temp = []
        for word in d:
            temp.append(dict.get(word, 0))
        train.append((1, temp))

    return train


################################################################
def get_data():
    dict = {}
    data_pos = []
    data_neg = []

    data_pos_path = '/home/zhangxin/文档/市场情绪分析/文献/依存句法/兰秋军/TotalCorpus/pos/seg_positive.txt'
    data_neg_path = '/home/zhangxin/文档/市场情绪分析/文献/依存句法/兰秋军/TotalCorpus/neg/seg_negative.txt'

    with open(data_pos_path) as f:
        flist = f.readlines()
        for d in flist:
            d = d.decode("gbk").strip("\n").replace(u"，", "").split(" ")
            data_pos.append(d)
            for dd in d:
                if not dd in dict:
                    dict[dd] = len(dict) + 1

    with open(data_neg_path) as f:
        flist = f.readlines()
        for d in flist:
            d = d.decode("gbk").strip("\n").replace(u"，", "").split(" ")
            data_neg.append(d)
            for dd in d:
                if not dd in dict:
                    dict[dd] = len(dict) + 1

    train = []
    for d in data_pos:
        temp = []
        for word in d:
            temp.append(dict.get(word, 0))
        train.append((1, temp))

    for d in data_neg:
        temp = []
        for word in d:
            temp.append(dict.get(word, 0))
        train.append((0, temp))

    return train


def get_data_test():
    get_data()
    data_pos = []

    data_pos_path = '/home/zhangxin/文档/市场情绪分析/文献/依存句法/兰秋军/TotalCorpus_test/test.txt'

    with open(data_pos_path) as f:
        flist = f.readlines()
        for d in flist:
            d = d.decode("utf-8").strip("\n").replace(u"，", "").split(" ")
            data_pos.append(d)

    train = []
    for d in data_pos:
        temp = []
        for word in d:
            temp.append(dict.get(word, 0))
        train.append((1, temp))

    return train


def load_data(n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    data = get_data()
    data = get_data_2()
    # data = get_data_test_2()
    train_set = []
    # test_set = []

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        # for x, y in zip(train_set[0], train_set[1]):
        for y, x in data:
            if len(x) < maxlen:
                # print (y, x)
                # print (type(x), type(y))
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    # test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)

    # test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sorted_index = len_argsort(test_set_x)
        # test_set_x = [test_set_x[i] for i in sorted_index]
        # test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    # test = (test_set_x, test_set_y)

    # return train, valid, test

    x = train[0]
    index_word = dict({(word_index_0[k], k) for k in word_index_0.keys()})
    # for i in index_word:
    #     print i, index_word[i]

    # for xx in x:
    # print xx
    # print " ".join([index_word[xxx] for xxx in xx])

    return train, valid


if __name__ == "__main__":
    train_data = get_data()
    for d in dict:
        if d.__contains__(u"，"):
            print d
        else:
            print d
