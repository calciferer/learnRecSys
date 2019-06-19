# 算法
# 从iter中去除指定元素后，随机挑选num个
import random


def randomSelect(iter, delete, num):
    l = iter.copy()
    for i in delete:
        if i in l :
            l.remove(i)
    random.shuffle(l)
    return l[:num]
