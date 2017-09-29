#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns


##############################################
# Global variables :
script_path = os.path.abspath(sys.argv[0])
working_dir = os.path.dirname(script_path)

data_dir = working_dir + "/data/"
data_reformated_dir = working_dir + "/reformated_data/"
results_dir = working_dir + "/results/"

cmap_orrd = ListedColormap(sns.color_palette("OrRd", 10).as_hex())


def get_dist_mat(df, target_col):
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Sorting according to clusters to make then apparent :
    M = np.concatenate((X, y[:, np.newaxis]), axis=1)
    # Sort according to last column :
    M = M[M[:, -1].argsort()]
    M = M[0: -1]  # remove last column

    from scipy.spatial.distance import pdist, squareform
    dist_mat = pdist(M, 'euclidean')
    dist_mat = squareform(dist_mat)  # translates this flattened form into a full matrix

    fig, ax = plt.subplots()
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
