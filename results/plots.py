#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'grid.color': 'k',
    'grid.linestyle': 'dashdot',
    'grid.linewidth': 0.6,
    'font.family': 'Linux Biolinum O',
    'font.size': 15,
    'axes.facecolor': 'white'
}
rcParams.update(params)


def PlotSummary(alexnet_workload,
                alexnet_accuracy,
                halfnet_workload,
                halfnet_accuracy,
                alexnet_s1,
                halfnet_s1,
                resnet):
    plt.figure()
    #plt.scatter(alexnet_workload[0], alexnet_accuracy[0], color='g', marker='x', linewidth='2')
    plt.plot(alexnet_workload, alexnet_accuracy, 'go-')
    plt.plot(halfnet_workload, halfnet_accuracy, 'b^-')
    plt.scatter(resnet[0], resnet[1], color='c', marker='x', linewidth='2')
    plt.scatter(alexnet_s1[0], alexnet_s1[1],
                color='m', marker='o', linewidth='1.5')
    plt.scatter(halfnet_s1[0], halfnet_s1[1],
                color='r', marker='^', linewidth='1.5')
    plt.legend(['MV-AlexNet', 'MVA-AlexNet-Half', 'ResNet', 'MV-AlexNet3-pool1', 'MV-AlexNet3-Half-pool1'])
    # plt.legend(['AlexNet', 'MV-AlexNet', 'MVA-AlexNet-Half', 'ResNet', 'MV-AlexNet3-pool1', 'MV-AlexNet3-Half-pool1'])
    plt.grid()
    plt.xlabel('Computational Workload (GMAC)')
    plt.ylabel('Top1 Accuracy (%)')
    plt.show()
    #plt.savefig("MVCNN-Perf.pdf", bbox_inches ='tight')


def PlotAcc(nb_views, alexnet_accuracy, halfnet_accuracy,
            alexnet_workload, halfnet_workload):
    width=0.4
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    plt.grid()
    plt.xlabel('Number of views')
    plt.xticks(nb_views, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.axis([0, 13, 81, 89])

    ax1.bar(nb_views-0.5*width, alexnet_accuracy, width, alpha= 0.6, edgecolor='black', color='b')
    ax1.bar(nb_views-0.5*width, halfnet_accuracy, width, alpha= 0.6, edgecolor='black', color='g')
    ax1.set_ylabel('Top1 Accuracy (%)')

    ax2 = ax1.twinx()
    ax2.bar(nb_views+0.5*width, alexnet_workload, width, alpha= 0.6, hatch="//", edgecolor='black', color='b')
    ax2.bar(nb_views+0.5*width, halfnet_workload, width, alpha= 0.6, hatch="//", edgecolor='black', color='g')
    ax2.set_ylabel('Workload (GMAC)')

    ax1.legend(['MVA', 'MVA-Half'])
    # ax2.legend(['MVA-Half', 'MVA'], loc='lower right', bbox_to_anchor=(0, 0.5))
    #plt.show()
    plt.savefig("NBView.pdf", bbox_inches ='tight')


if __name__ == '__main__':
    nb_views         = np.array([1, 2, 3, 4, 11, 12])
    nb_views_all     = np.linspace(1,12,12,endpoint=True)
    alexnet_accuracy = np.array([85.3, 87.4, 87.9, 88, 88.5, 88.6])
    alexnet_workload = np.array([0.67, 1.33, 2, 2.66, 7.32, 7.99])
    halfnet_accuracy = np.array([81.8, 84.55, 85.2, 85.55, 86.65, 86.65])
    halfnet_workload = np.array([0.14, 0.39, 0.58, 0.77, 2.12, 2.31])
    halfnet_s1       = np.array([0.30, 84.95])
    alexnet_s1       = np.array([0.87, 87.3])
    resnet           = np.array([3.86, 87.1])
    alexnet_accd     = np.interp(nb_views_all, nb_views, alexnet_accuracy)
    halfnet_accd     = np.interp(nb_views_all, nb_views, halfnet_accuracy)
    alexnet_word     = 0.67 * nb_views_all
    halfnet_word     = 0.14 * nb_views_all
    PlotAcc(nb_views_all, alexnet_accd, halfnet_accd, alexnet_word, halfnet_word)

    # PlotSummary(alexnet_workload = alexnet_workload,
    #             halfnet_workload = halfnet_workload,
    #             alexnet_accuracy = alexnet_accuracy,
    #             halfnet_accuracy = halfnet_accuracy,
    #             alexnet_s1 = alexnet_s1,
    #             halfnet_s1 = halfnet_s1,
    #             resnet = resnet)
