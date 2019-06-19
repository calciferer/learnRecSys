# coding=utf-8
"""
评价指标
"""
import math


def recall(data):
    """
    召回率
    :param data格式为[({tu},{ru}),({},{})...]
    :param tuList:测试集物品列表[{},{},...]
    :param ruList:推荐的物品列表[{},{},...]
    公式:对于每个u求和 len(ru & tu)/对于每个u求和len(tu)
    """
    hit = 0
    all = 0
    for tu, ru in data:
        hit += len(tu & ru)
        all += len(tu)
    return hit / float(all)


def precision(data):
    """
    召回率
    :param data格式为[({tu},{ru}),({},{})...]
    :param tuList:测试集物品列表[{},{},...]
    :param ruList:推荐的物品列表[{},{},...]
    公式:对于每个u求和 len(ru & tu)/对于每个u求和len(ru)
    """
    hit = 0
    all = 0
    for tu, ru in data:
        hit += len(tu & ru)
        all += len(ru)
    return hit / float(all)


def coverage(allRu: set, trainItems: set) -> float:
    """
    :param allRu: 所有的ru
    :param trainItems: 训练集的所有item
    """
    # print(allRu)
    # print(trainItems)
    return len(allRu) / len(trainItems)


def popularity(trainSet, ruList):
    """
    流行度
    """
    itemPopularity = {}
    for userId, itemIds in trainSet.items():
        for itemId in itemIds:
            itemPopularity.setdefault(itemId, 0)
            itemPopularity[itemId] += 1

    sum = 0
    n = 0
    for ru in ruList:
        for itemId in ru:
            sum += math.log(1 + itemPopularity[itemId])
        n += len(ru)

    return sum / n
