import logging
from operator import itemgetter

import math

from abs.AbsRecommender import AbsRecommender
from aop.LogAop import log
from commen import SplitData, Evaluation
from util import FileUtil


class ItemCF(AbsRecommender):
    __rawData = []
    __dataSet = {}
    __trainSet = {}
    __testSet = {}
    __W = {}
    __recResult = {}

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
        pass

    @log()
    def calcSimMatrix(self):
        """
        计算item相似度矩阵w
        """
        # logging.info("生成物品相似度矩阵")
        # self.__timer.start()
        # 得到所有训练集物品
        allItems = set()

        for userId, itemIds in self.__trainSet.items():
            allItems.update(itemIds)
        # logging.info('一共有：%d个物品', len(allItems))
        # 1.计算C[i][j]
        C = self.__initMatrix(allItems, 0)
        N = {}
        for userId, itemIds in self.__trainSet.items():
            for i in itemIds:
                N.setdefault(i, 0)
                N[i] += 1
                for j in itemIds:
                    C[i][j] += 1
        # logging.debug('C[u][v]:%s', str(C))
        # 2.计算W[i][j]
        self.__W = self.__initMatrix(allItems, 0)
        for i, ralatedItemIds in C.items():
            for j, cij in ralatedItemIds.items():
                self.__W[i][j] = C[i][j] / math.sqrt(N[i] * N[j])
        # logging.debug('W[i][j]:%s:', str(self.__W))
        # logging.info('生成物品相似度矩阵完成,用时:%dms', self.__timer.stop())
        return self.__W

    @log()
    def recommendForAllUser(self, K):
        """
        生成训练集上所有用户的推荐列表,计算每个用户的item列表中每个item最相似的K个item的兴趣度puj
        公式：p(u,j) = 求和W[j][i]*R[u][i],对所有的i属于S(j,K) & N(u)，这里用隐反馈，R[u][i]=1
        :param K: 最相近的K个用户
        :return: 每个用户的推荐结果
        """
        # logging.info("为所有用户生成推荐")
        # self.__timer.start()
        for u in self.__trainSet.keys():
            self.__recResult[u] = {}
            ru = self.__trainSet[u]
            for i in ru:
                for j, wij in sorted(self.__W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
                    if j not in ru:
                        self.__recResult[u].setdefault(j, 0.0)
                        self.__recResult[u][j] += self.__W[i][j]

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

    @log()
    def __initMatrix(self, indexes, initialValue):
        # 初始化矩阵
        w = {}
        for index1 in indexes:
            w[index1] = {}
            for index2 in indexes:
                w[index1][index2] = initialValue
        return w
