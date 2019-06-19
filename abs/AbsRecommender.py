# 定义推荐规范
import time
from abc import ABCMeta, abstractmethod


class AbsRecommender(metaclass=ABCMeta):

    @abstractmethod
    def loadRawData(self, path):
        """
        加载原始数据
        :param path: 文件路径
        """

        pass

    @abstractmethod
    def generateDataset(self):
        """
        生成训练集和测试集
        """
        pass

    @abstractmethod
    def calcSimMatrix(self):
        """
        计算相似度矩阵
        """
        pass

    @abstractmethod
    def recommendForAllUser(self, K):
        """
        为每一个用户生成推荐列表
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        评估
        """
        pass
