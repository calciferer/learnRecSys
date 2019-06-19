from operator import itemgetter

import math

from abs.AbsRecommender import AbsRecommender
from aop.LogAop import log
from commen import SplitData, Evaluation
from util import FileUtil
import logging


class UserCF(AbsRecommender):
    __rawData = []
    __dataSet = {}
    __trainSet = {}
    __testSet = {}
    __W = {}
    __recResult = {}
    __itemId2UserIds = {}

    def __init__(self):
        pass

    @log()
    def __initMatrix(self, indexes, initialValue):
        # 初始化矩阵
        w = {}
        for index1 in indexes:
            w[index1] = {}
            for index2 in indexes:
                w[index1][index2] = initialValue
        return w

    @log()
    def loadRawData(self, path) -> []:
        # logging.info("开始加载原始数据,path = %s", path)
        # self.__timer.start()
        self.__rawData = FileUtil.readAllLines(path)
        self.__rawData.pop(0)  # 删除第一行的title
        # logging.debug("原始数据长度:%d,原始数据:%s", len(self.__rawData), str(self.__rawData))
        # logging.info('加载数据完成,用时:%dms', self.__timer.stop())
        return self.__rawData

    @log()
    def generateDataset(self) -> ({str: [()]}, {str: [()]}):
        # logging.info("生成dataSet,trainSet,testSet,不考虑评分和时间戳")
        # self.__timer.start()
        for line in self.__rawData:
            split = line.split(',')
            userId = split[0]
            itemId = split[1]
            self.__dataSet.setdefault(userId, set())
            self.__dataSet[userId].add(itemId)
        # logging.debug("dataSet:长度：%d,数据：%s", len(self.__dataSet), str(self.__dataSet))
        # logging.info("dataSet:长度：%d", len(self.__dataSet))
        # 将每个userId对应的items分割,80%作训练集,20%作测试集
        for userId, itemIds in self.__dataSet.items():
            testSetItems, trainSetItems = SplitData.randomSplitList(list(itemIds), 0.2)
            self.__trainSet[userId] = set(trainSetItems)
            self.__testSet[userId] = set(testSetItems)
        # logging.info("trainSet:长度：%d", len(self.__trainSet))
        # logging.debug("trainSet:长度：%d,数据：%s", len(self.__trainSet), str(self.__trainSet))
        # logging.debug("testSet:长度：%d,数据：%s", len(self.__testSet), str(self.__testSet))
        # logging.info("testSet:长度：%d", len(self.__testSet))
        # logging.info('生成数据集完成,用时:%dms', self.__timer.stop())

    @log()
    def calcSimMatrix(self) -> {}:
        """
        计算user相似度矩阵w
        w[u][v]:用户u和v的相似度
        N(u):用户u的物品列表
        N(v):用户v的物品列表
        C[u][v]用户u和v共有的物品数 即 c[u][v] = len(N(u) & N(v))
        公式：w[u][v] = c[u][v]/sqrt(len(N(u))*len(N(v)))
        """
        # logging.info("生成用户相似度矩阵")
        # self.__timer.start()
        # 1.建立item-user倒排表
        itemId2UserIds = {}
        for userId, itemIds in self.__trainSet.items():
            for itemId in itemIds:
                itemId2UserIds.setdefault(itemId, set())
                itemId2UserIds[itemId].add(userId)
        # logging.debug('item-user倒排表：%s', str(itemId2UserIds))
        self.__itemId2UserIds = itemId2UserIds
        # 2.计算C[u][v]
        C = self.__initMatrix(self.__trainSet.keys(), 0)
        for itemId, userIds in itemId2UserIds.items():
            for u in userIds:
                for v in userIds:
                    C[u][v] += 1
        # logging.debug('C[u][v]:%s', str(C))
        # 3.计算W[u][v]
        self.__W = self.__initMatrix(self.__trainSet.keys(), 0)
        for u in self.__trainSet.keys():
            for v in self.__trainSet.keys():
                self.__W[u][v] = C[u][v] / math.sqrt(len(self.__trainSet[u]) * len(self.__trainSet[v]))
        # logging.debug('W[u][v]:%s:', str(self.__W))
        # logging.info('生成用户相似度矩阵完成,用时:%dms', self.__timer.stop())
        return self.__W

    @log()
    def recommendForAllUser(self, K):
        """
        生成训练集上所有用户的推荐列表,计算每个用户和他相邻的K个用户的物品列表中的每个物品的兴趣程度pui
        公式：p(u,i) = 求和W[u][v]*R[v][i],对所有的v属于S(u,K) & N(i)，这里用隐反馈，R[v][i]=1
        :param K: 最相近的K个用户
        :return: 每个用户的推荐结果
        """
        # logging.info("为所有用户生成推荐")
        # self.__timer.start()
        for u in self.__trainSet.keys():
            self.__recResult[u] = {}
            for v, wuv in sorted(self.__W[u].items(), key=itemgetter(1), reverse=True)[0:K]:
                for itemId in self.__trainSet[v]:
                    if itemId not in self.__trainSet[u]:
                        self.__recResult[u].setdefault(itemId, 0.0)
                        self.__recResult[u][itemId] += self.__W[u][v]

        # logging.info('为所有用户生成推荐完成,用时:%dms', self.__timer.stop())
        # logging.info('用户1的推荐列表%s', sorted(self.__recResult['1'].items(), key=itemgetter(1), reverse=True))
        return self.__recResult

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

        trainItems = set(self.__itemId2UserIds.keys())
        coverage = Evaluation.coverage(allRu, trainItems)

        popularity = Evaluation.popularity(self.__trainSet, ruList)

        # logging.info('评估指标完成,用时:%dms', self.__timer.stop())
        # logging.info('召回率：%.2f%%', recall * 100)
        # logging.info('准确率：%.2f%%', precision * 100)
        # logging.info('覆盖率：%.2f%%', coverage * 100)
        # logging.info('流行度：%.2f', popularity)
        return recall, precision,coverage,popularity
