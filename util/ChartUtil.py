import numpy as np
import matplotlib.pyplot as plt


def drawLineChart(title,xLabel, yLabel, lineDatas):
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for lineData in lineDatas:
        plt.plot(lineData[0], lineData[1], label=lineData[2])

    plt.xlabel(xLabel)

    plt.ylabel(yLabel)

    plt.title(title)

    plt.legend()

    plt.show()
