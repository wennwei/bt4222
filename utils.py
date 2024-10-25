import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from pylab import figure, show, legend, ylabel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import warnings
from datetime import datetime
from itertools import combinations
from collections import defaultdict
import itertools
import os
import sys
import networkx as nx
import math

def numerical_feature_dist(df, feature, by):
    print("Describe", feature)
    df = df.sort_values(by)
    ax = sns.boxplot(x=by, y=feature, data=df, color='white', width=.5)
    axes = plt.gca()
    axes.set_ylim([min(df[feature]), max[df[feature]]])
    plt.xticks(rotation = 90)
    plt.show()
    
    if by == None:
        plt.xlabel('')
        return df[feature].describe().to_frame().T
    else:
        return pd.concat([
            df[feature].describe().to_frame().T,
            df.groupby(by)[feature].describe
        ])
