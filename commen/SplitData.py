# coding=utf-8
"""
分割数据
"""
import random

import math


def randomSplitList(listData: [], ratio: float) -> ([], []):
    """
    按比例分割list
    :rtype: ([],[])
    :param listData:
    :param ratio: 分割比例
    """
    list1 = []
    list2 = []
    # 将list乱序,然后从中挑出ratio*len(listData)个元素放入list1,剩下的放入list2
    listCopy = listData.copy()
    random.shuffle(listCopy)
    n = math.ceil(ratio*len(listCopy))
    list1 = listCopy[:n]
    list2 = listCopy[n:]
    return list1, list2
