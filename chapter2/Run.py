import logging.config
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from abs.RecommenderFactory import RecommenderFactory
from chapter2.ItemCF import ItemCF
from chapter2.LFM import LFM
from chapter2.UserCF import UserCF
from util import ChartUtil, FileUtil

if __name__ == '__main__':
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.config.fileConfig(os.getcwd() + '/../config/logging.config', disable_existing_loggers=False)

    Ks = [10]
    recalls = []
    precisions = []
    coverages = []
    popularitys = []

    for K in Ks:
        print(K)
        recommender = RecommenderFactory.getRecommender(RecommenderFactory.LFM)
        recommender.loadRawData(os.getcwd() + '/../dataset/movielens/ml-latest-small/ratings.csv')
        recommender.generateDataset()
        recommender.calcSimMatrix()
        recommender.recommendForAllUser(K)
        recall, precision, coverage, popularity = recommender.evaluate()
        recalls.append(recall)
        precisions.append(precision)
        coverages.append(coverage)
        popularitys.append(popularity)

    data = [(np.asarray(Ks), np.asarray(recalls), 'recall'), (np.asarray(Ks), np.asarray(precisions), 'precisions'),
            (np.asarray(Ks), np.asarray(coverages), 'coverages')]
    ChartUtil.drawLineChart('userCF-evaluation', 'K', 'Value', data)
    ChartUtil.drawLineChart('userCF-popularity', 'K', 'Value', [
        (np.asarray(Ks), np.asarray(popularitys), 'popularitys')])
