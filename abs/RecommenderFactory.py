# 推荐器工厂
from chapter2.ItemCF import ItemCF
from chapter2.LFM import LFM
from chapter2.UserCF import UserCF


class RecommenderFactory:
    UserCF = 'UserCF'
    ItemCF = 'ItemCF'
    LFM = 'LFM'

    @staticmethod
    def getRecommender(name, **kvs):
        if name == RecommenderFactory.UserCF:
            return UserCF()
        elif name == RecommenderFactory.ItemCF:
            return ItemCF()
        elif name == RecommenderFactory.LFM:
            return LFM()
        else:
            raise Exception('No Such Recommender')
