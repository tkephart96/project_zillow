'''Explore Zillow data

Functions:
- pear
- baseline
'''

########## IMPORTS ##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from scipy import stats


########## EXPLORE ##########

def pear(train, x, y, alt_hyp='two-sided'):
    '''Spearman's R test with a print'''
    r,p = stats.spearmanr(train[x], train[y], alternative=alt_hyp)
    print(f"Spearman's R: {x} and {y}\n", f'r = {r}, p = {p}')

def baseline(target):
    """
    The function calculates and prints the accuracy of a baseline model that always predicts the most
    frequent class in the target variable.
    
    :param target: The "target" parameter is likely a Pandas Series or DataFrame column that contains
    the true labels or values that we are trying to predict or classify. The "baseline" function appears
    to calculate the accuracy of a simple baseline model that always predicts the most common value in
    the "target" column
    """
    print(f'Baseline Property Value: {round(((target).mean()),2)}')
