# general
import json
import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pipeline and ML stuff
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
# Clustering
from sklearn.cluster import KMeans

from collections import OrderedDict
from collections import Counter
import shap

# Neural Network Utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# # Set the random seed
# np.random.seed(123)
# random_state = 123

# Add the evals outer directory
sys.path.append('../evals')
from evals.eval_surv_copy import EvalSurv

#_______________________________________________________________________________________________
# Reusable Functions
def plot_cluster_hist(arr: np.array):
    '''
    function  to sho histograms of clusters
    '''
    cluster_counts = Counter(arr)
    # Sort the Counter object by keys in ascending order
    sorted_countes = OrderedDict(sorted(cluster_counts.items()))
    
    # Extract the category labels and their corresponding counts
    labels = list(sorted_countes.keys())
    counts = list(sorted_countes.values())

    # Plot the histogram
    plt.bar(labels, counts);

    # Set the labels and title
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Categorical Histogram')
    plt.xticks(labels)

    for i, value in enumerate(counts):
        plt.text(i, value + 1, str(value), ha='center', va='bottom')

    plt.show() 

def plot_with_cf(bin_edges , mean_ , low_ , up_ , _from , _to = None , transparency = 0.05):
    '''
    function to plot survival times with cf
    '''
    if _to is not None:
        for _ , (m , l , u) in enumerate(zip(mean_[_from:_to] , low_[_from:_to] , up_[_from:_to])):
            plt.step(bin_edges , m , where = 'post' , label = 'mean');
            plt.fill_between(bin_edges , l , u , step = 'post' , alpha = transparency , label = 'confint');
    else:
        m , l , u = mean_[_from] , low_[_from] , up_[_from]
        plt.step(bin_edges , m , where = 'post' , label = 'mean');
        plt.fill_between(bin_edges , l , u , step = 'post' , alpha = transparency , label = 'confint');
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    
    # Save the plot as an SVG file
    plt.savefig('graph.pdf', format='pdf')

    # plt.show();

def get_target(df):
    '''
    function to get survival time and survival status
    '''
    return df['time_to_event'].values, df['death'].values
#___________________________________________________________________________________________________
# Configs
config_file_path = "config.json"