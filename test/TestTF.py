# import logging

# from algo import Algo
# from aop import LogAop
# from aop.LogAop import log
# import numpy as np
import os
import sys
import tensorflow as tf


# @log()
# def func1(parama, paramb):
#     return 'abc', 'dfg'
#     pass
#
#
# @log()
# def func2(parama, paramb):
#     return 'abc', 'dfg'
#     pass


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG,
    #                     format="%(asctime)s-%(levelname)s-%(module)s-%(funcName)s-%(lineno)s\n%(message)s\n")

    # a = np.array([1, 2, 3], np.int8)
    # print(sys.getsizeof(a))
    # a = np.array([(1, 1.2, 's'), (2, 1.3, 'k')], dtype=np.dtype([('a', '<i1'), ('b', '<f2'), ('c', 'U1')]))
    # a = np.array([1,2,3,4],dtype=np.dtype('<u1'))
    # print(a)
    # print(a.dtype)
    # print(sys.getsizeof(a))
    # print(sys.getsizeof([1,2,3]))
    # print(sys.getsizeof([1,2,3,4]))
    # a = np.array([[1, 2], [3, 4]],dtype=np.dtype('u1'))
    # print(a)
    # print(a.ndim)
    # print(a.shape)
    # print(a.size)
    # print(a.itemsize)
    # print(a.flags)
    # print(a.real)
    # print(a.imag)
    # print(a.data)
    # a = np.empty([3, 2], dtype=np.dtype('<u1'))
    # print(a)\
    # a = np.frombuffer(b'hello',dtype='S1')
    # a = np.fromiter({1,2,3},dtype=int)
    # print(a)
    # x = np.array([[1, 2], [3, 4], [5, 6]])
    # y = x[[0, 1, 2], [0, 1, 0]]
    # print(x)
    # print(y)
    # print(x[0][0])
    # print(x[0,0])
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(a)
    # b = a[1:3, 1:3]
    # c = a[1:3, [1, 2]]
    # d = a[..., 1:]
    # print(b)
    # print(c)
    # print(d)
    # a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
    # print(a)
    # print(a[..., 1])  # 第2列元素
    # print(a[1, ...])  # 第2行元素
    # print(a[..., 1:])  # 第2列及剩下的所有元素
    # x = np.arange(32).reshape((8, 4))
    # print(x)
    # print(x[...,[3,  0]])
    # print(x[np.ix_([0, 1, 2, 3], [0, 1, 2, 2])])
    # a = np.arange(0, 60, 5)
    # a = a.reshape(3, 4)
    # print('原始数组是：')
    # print(a)
    # print('\n')
    # for x in np.nditer(a, flags=['multi_index']):
    #     print(x, end=", ")
    # for row in a:
    #     print(row)
    # print(a.flat)
    # for x in a.flat:
    #     print(x)
    # a = np.arange(18).reshape(2,3,3)
    # print(a)
    # print('---')
    # print(np.squeeze(a,2))
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[5, 6], [7, 8]])
    # print(a.ndim)
    # print(a)
    # print(b)
    # c = np.concatenate((a,b),axis=0)
    # print(c)
    # a = np.arange(16).reshape(4, 4)
    #
    # print('第一个数组：')
    # print(a)
    # print('\n')
    #
    # print('竖直分割：')
    # b = np.vsplit(a, 2)
    # print(b)
    # a = np.asarray(['1,a','2,b'])
    # print(a)
    # b = np.char.split(a,',')
    # c = []
    # for row in b:
    #     c.append(row)
    # c = np.asarray(c)
    # print(c)
    # a = np.arange(10).reshape(2,5)
    # print(a)
    # print(np.amax(a,axis=1))
    # a = np.asarray([(20,100,'a'),(21,101,'b')])
    # np.lexsort(a)
    # print(a)
    # func1(1, 2)
    # func2(1, 2)
    # print(Algo.randomSelect([1, 2, 3, 4, 56, 78], ['a'], 5))
    # a = {'a':1,'b':2,'c':3}
    # print(set(a.values()))
    # print(type(a.values()))
    # print(np.square(list(a.values())))
    print("hello123")
    print(os.getcwd())
    hello = tf.constant('hello,tensorf')
    sess = tf.Session()
    print(sess.run(hello))
    pass
