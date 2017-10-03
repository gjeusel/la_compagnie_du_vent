#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls


##############################################
# Global variables :
script_path = os.path.abspath(sys.argv[0])
working_dir = os.path.dirname(script_path)

data_dir = working_dir + "/data/"
data_reformated_dir = working_dir + "/reformated_data/"
results_dir = working_dir + "/results/"

cmap_orrd = ListedColormap(sns.color_palette("OrRd", 10).as_hex())

cmap_orrd = ListedColormap(sns.color_palette("OrRd", 10).as_hex())


def get_dist_mat(df, target_col, metric='euclidean', figsize=(20,20)):
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Sorting according to clusters to make then apparent :
    M = np.concatenate((X, y[:, np.newaxis]), axis=1)
    # Sort according to last column :
    M = M[M[:, -1].argsort()]
    M = M[0: -1]  # remove last column

    from scipy.spatial.distance import pdist, squareform
    dist_mat = pdist(M, metric=metric)
    dist_mat = squareform(dist_mat)  # translates this flattened form into a full matrix

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_mat, cmap=cmap_orrd, interpolation='none')

    # get colorbar smaller than matrix
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # want a more natural, table-like display
    ax.invert_yaxis()

    # Move top xaxes :
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.axis('off')

    return dist_mat, fig, ax


def get_corr_mat(df, figsize=(20, 20)):
    # Compute correlation matrix
    corrmat = df.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set ax & colormap with seaborn.
    ax = sns.heatmap(corrmat, vmin=-1, vmax=1, center=0,
                     square=True, linewidths=1, xticklabels=True,
                     yticklabels=True)

    ax.set_xticklabels(df.columns, minor=False, rotation='vertical')
    ax.set_yticklabels(df.columns[df.shape[1]::-1], minor=False, rotation='horizontal')

    return corrmat, fig, ax


def get_scatter_mat(df, figsize=(20,20)):
    from pandas.plotting import scatter_matrix
    axs = scatter_matrix(df, alpha=0.5, figsize=figsize)

    for ax in axs[:,0]: # the left boundary
        # ax.grid('off', axis='both')
        ax.set_ylabel(ax.get_ylabel(), rotation=0, verticalalignment='center', labelpad=55)
        ax.set_yticks([])

    return axs


# def get_distrib_barplot(pdSerie, color='#40466e', figsize=(20, 20)):
#     ax = pdSerie.hist(
#         color=color, alpha=0.8, bins=20, figsize=figsize)
#     return ax


def horizontal_boxplot(df):
    sns.set(style="ticks")
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(20, 20))
    ax.set_xscale("log")

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="distance", y="feature", data=df,
                whis=np.inf, palette="vlag")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    return f, ax


def get_distrib_barplot(df):

    pass

# def get_pca(X, y, n_components):
#     from sklearn.decomposition import PCA

#     pca = PCA(n_components=n_components)
#     X_r = pca.fit(X).transform(X)



#     pca = PCA(n_components=2)
#     X_r = pca.fit(X).transform(X)

#     lda = LinearDiscriminantAnalysis(n_components=2)
#     X_r2 = lda.fit(X, y).transform(X)

#     # Percentage of variance explained for each components
#     print('explained variance ratio (first two components): %s'
#         % str(pca.explained_variance_ratio_))

#     plt.figure()
#     colors = ['navy', 'turquoise', 'darkorange']
#     lw = 2

#     for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#         plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
#                     label=target_name)
#     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title('PCA of IRIS dataset')

#     plt.figure()
#     for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#         plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
#                     label=target_name)
#     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title('LDA of IRIS dataset')

#     return

