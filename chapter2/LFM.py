import logging
import random
from operator import itemgetter

from abs.AbsRecommender import AbsRecommender
from algo import Algo
from aop.LogAop import log
from commen import SplitData, Evaluation
from util import FileUtil
import numpy as np


class LFM(AbsRecommender):
    __rawData = []
    __dataSet = {}
    __trainSet = {}
    __testSet = {}
    __recResult = {}
    __trainSet2 = {}

    __F = 10
    __alpha = 0.02
    __lambda = 0.01

    __P = {}
    __Q = {}

    def __init__(self, F=10, alpha=0.02, _lambda=0.01):
        self.__F = F
        self.__alpha = alpha
        self.__lambda = _lambda

    def __predict(self, user, item):
        pui = 0.0
        for f in range(self.__F):
            pui += self.__P[user][f] * self.__Q[item][f]
        return pui

    @log()
    def loadRawData(self, path):
        # logging.info("开始加载原始数据,path = %s", path)
        # self.__timer.start()
        self.__rawData = FileUtil.readAllLines(path)
        self.__rawData.pop(0)  # 删除第一行的title
        # logging.debug("原始数据长度:%d,原始数据:%s", len(self.__rawData), str(self.__rawData))
        # logging.info('加载数据完成,用时:%dms', self.__timer.stop())
        return self.__rawData
        pass

    @log()
    def generateDataset(self):
        # 生成训练集和测试集
        for line in self.__rawData:
            split = line.split(',')
            userId = split[0]
            itemId = split[1]
            self.__dataSet.setdefault(userId, set())
            self.__dataSet[userId].add(itemId)
        for userId, itemIds in self.__dataSet.items():
            testSetItems, trainSetItems = SplitData.randomSplitList(list(itemIds), 0.2)
            self.__trainSet[userId] = set(trainSetItems)
            self.__testSet[userId] = set(testSetItems)
        # 在训练集中,生成用户的正样本和负样本
        # 正样本为用户已经产生行为的物品
        # 负样本为在整个物品列表中随机挑选用户没有产生过的物品,保证正负样本数量相同
        itemsPool = []  # 所有物品,流行度高的出现的次数多
        for user, items in self.__trainSet.items():
            itemsPool.extend(items)

        # 带正负样本的训练集
        for user, items in self.__trainSet.items():
            newItems = {}
            for item in items:
                newItems[item] = 1
            # 从itemsPool中随机挑选len(items)个负样本
            negativeItems = Algo.randomSelect(itemsPool, items, len(items))
            for negativeItem in negativeItems:
                newItems[negativeItem] = 0
            self.__trainSet2[user] = newItems

    @log()
    def calcSimMatrix(self):
        # 初始化P,Q
        # P = {}  # user*f
        # Q = {}  # item*f
        for user, items in self.__trainSet.items():
            self.__P[user] = {}
            for f in range(self.__F):
                self.__P[user][f] = random.random()
        allItems = set()
        for user, items in self.__trainSet.items():
            allItems.update(items)
        for item in allItems:
            self.__Q[item] = {}
            for f in range(self.__F):
                self.__Q[item][f] = random.random()
        # self.__P = P
        # self.__Q = Q
        # 训练过程
        for step in range(10):
            for user, items in self.__trainSet2.items():
                for item, rui in items.items():
                    eui = rui - self.__predict(user, item)
                    for f in range(self.__F):
                        self.__P[user][f] += self.__alpha * (
                                    eui * self.__Q[item][f] - self.__lambda * self.__P[user][f])
                        self.__Q[item][f] += self.__alpha * (
                                    eui * self.__P[user][f] - self.__lambda * self.__Q[item][f])
            self.__alpha *= 0.9
            loss = self.loss()
            print('loss:%.6f' % loss)

    @log()
    def recommendForAllUser(self):
        for user, items in self.__trainSet.items():
            for item in items:
                pui = self.__predict(user, item)
                self.__recResult[user][item] = pui

    @log()
    def evaluate(self):
        """
                评估推荐效果，ru：为u推荐的itemList,tu:测试集上u的itemList
                """
        # logging.info("开始评估指标")
        # self.__timer.start()
        data = []
        allRu = set()
        ruList = []
        for userId, recItems in self.__recResult.items():
            tu = self.__testSet[userId]
            ru = {itemId for itemId, pui in sorted(recItems.items(), key=itemgetter(1), reverse=True)[0:10]}
            ruList.append(ru)
            allRu.update(ru)
            data.append((tu, ru))
        recall = Evaluation.recall(data)
        precision = Evaluation.precision(data)

        trainItems = set()
        for userId, itemIds in self.__trainSet.items():
            trainItems.update(itemIds)
        coverage = Evaluation.coverage(allRu, trainItems)

        popularity = Evaluation.popularity(self.__trainSet, ruList)

        # logging.info('评估指标完成,用时:%dms', self.__timer.stop())
        # logging.info('召回率：%.2f%%', recall * 100)
        # logging.info('准确率：%.2f%%', precision * 100)
        # logging.info('覆盖率：%.2f%%', coverage * 100)
        # logging.info('流行度：%.2f', popularity)
        return recall, precision, coverage, popularity

    def loss(self):
        C = 0.
        for user, user_latent in self.__P.items():
            for item, item_latent in self.__Q.items():
                rui = 0.0
                try:
                    rui = self.__trainSet2[user][item]
                except:
                    rui = 0.0
                eui = rui - self.__predict(user, item)
                C += (np.square(eui) +
                      self.__lambda * np.sum(np.square(list(self.__P[user].values()))) +
                      self.__lambda * np.sum(np.square(list(self.__Q[item].values())))
                      )
        return C
