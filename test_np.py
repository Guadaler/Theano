#!/usr/bin/env python
# encoding: utf-8

# -------------------------------------------
# 功能：
# Author: zx
# Software: PyCharm Community Edition
# File: test_np.py
# Time: 17-1-3 上午10:27
# -------------------------------------------

import numpy as np

a = np.random.randn(3, 3)
print a.ndim
print a.__len__()
print a.shape[0]
print a.shape[1]
print a.shape

b = np.random.rand(5, 4)
c = [1, 2]
d = b[c]
print b
print d
