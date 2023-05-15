'''Model Zillow data

Functions:
- metrics_reg
- reg_mods
'''

########## IMPORTS ##########
import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

######### FUNCTIONS #########

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def reg_mods(Xtr,ytr,Xv,yv,features=None):
    '''
    send in X_train,y_train,X_val,y_val,and list of features
    so that function will run through linear regression, lasso lars,
    polynomial feature regression, and tweedie regressor (glm)
    - diff feature combos
    - output as df
    '''
    if features is None:
        features = Xtr.columns.to_list()
    # baseline as mean
    pred_mean = ytr.mean()[0]
    ytr_p = ytr.assign(pred_mean=pred_mean)
    yv_p = yv.assign(pred_mean=pred_mean)
    rmse_tr = mean_squared_error(ytr,ytr_p.pred_mean)**.5
    rmse_v = mean_squared_error(yv,yv_p.pred_mean)**.5
    r2_tr = r2_score(ytr, ytr_p.pred_mean)
    r2_v = r2_score(yv, yv_p.pred_mean)
    output = {
            'model':'bl_mean',
            'features':'None',
            'params':'None',
            'rmse_tr':rmse_tr,
            'rmse_v':rmse_v,
            'r2_tr':r2_tr,
            'r2_v':r2_v
        }
    metrics = [output]
    # baseline as median
    pred_median = ytr.median()[0]
    ytr_p,yv_p=ytr,yv
    ytr_p = ytr_p.assign(pred_median=pred_median)
    yv_p = yv_p.assign(pred_median=pred_median)
    rmse_tr = mean_squared_error(ytr,ytr_p.pred_median)**.5
    rmse_v = mean_squared_error(yv,yv_p.pred_median)**.5
    r2_tr = r2_score(ytr, ytr_p.pred_median)
    r2_v = r2_score(yv, yv_p.pred_median)
    output = {
            'model':'bl_median',
            'features':'None',
            'params':'None',
            'rmse_tr':rmse_tr,
            'rmse_v':rmse_v,
            'r2_tr':r2_tr,
            'r2_v':r2_v
        }
    metrics.append(output)
    # create iterable for feature combos
    for r in range(1,(len(features)+1)):
        # print(r)
        # cycle through feature combos for linear reg
        # print('start lin reg')
        for feature in itertools.combinations(features,r):
            f = list(feature)
            # linear regression
            lr = LinearRegression()
            lr.fit(Xtr[f],ytr)
            # metrics
            pred_lr_tr = lr.predict(Xtr[f])
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lr_tr)
            pred_lr_v = lr.predict(Xv[f])
            rmse_v,r2_v = metrics_reg(yv,pred_lr_v)
            # table-ize
            output ={
                    'model':'LinearRegression',
                    'features':f,
                    'params':'None',
                    'rmse_tr':rmse_tr,
                    'r2_tr':r2_tr,
                    'rmse_v':rmse_v,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos and alphas for lasso lars
        # print('start lasso lars')
        for feature in itertools.combinations(features,r):
            f = list(feature)
            # lasso lars
            ll = LassoLars(alpha=1,normalize=False)
            ll.fit(Xtr[f],ytr)
            # metrics
            pred_ll_tr = ll.predict(Xtr[f])
            rmse_tr,r2_tr = metrics_reg(ytr,pred_ll_tr)
            pred_ll_v = ll.predict(Xv[f])
            rmse_v,r2_v = metrics_reg(yv,pred_ll_v)
            # table-ize
            output ={
                    'model':'LassoLars',
                    'features':f,
                    'params':'alpha=1',
                    'rmse_tr':rmse_tr,
                    'r2_tr':r2_tr,
                    'rmse_v':rmse_v,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos and degrees for polynomial feature reg
        # print('start poly reg')
        for feature,d in itertools.product(itertools.combinations(features,r),[3,4]):
            f = list(feature)
            # polynomial feature regression
            pf = PolynomialFeatures(degree=d)
            Xtr_pf = pf.fit_transform(Xtr[f])
            Xv_pf = pf.transform(Xv[f])
            lp = LinearRegression()
            lp.fit(Xtr_pf,ytr)
            # metrics
            pred_lp_tr = lp.predict(Xtr_pf)
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lp_tr)
            pred_lp_v = lp.predict(Xv_pf)
            rmse_v,r2_v = metrics_reg(yv,pred_lp_v)
            # table-ize
            output ={
                    'model':'PolynomialFeature',
                    'features':f,
                    'params':f'degree={d}',
                    'rmse_tr':rmse_tr,
                    'r2_tr':r2_tr,
                    'rmse_v':rmse_v,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos, alphas, and powers for tweedie reg
        # print('start tweedie reg')
        for feature,a in itertools.product(itertools.combinations(features,r),[1,2]):
            f = list(feature)
            # print(f,' - ',a)
            # tweedie regressor glm
            lm = TweedieRegressor(power=2,alpha=a)
            # print('model made')
            lm.fit(Xtr[f],ytr.prop_value)
            # print('model fit')
            # metrics
            pred_lm_tr = lm.predict(Xtr[f])
            # print('pred tr made',pred_lm_tr.max())
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lm_tr)
            # print('metric tr made')
            pred_lm_v = lm.predict(Xv[f])
            # print('pred v made')
            rmse_v,r2_v = metrics_reg(yv,pred_lm_v)
            # print('metric v made')
            # table-ize
            output ={
                    'model':'TweedieRegressor',
                    'features':f,
                    'params':f'power=2,alpha={a}',
                    'rmse_tr':rmse_tr,
                    'r2_tr':r2_tr,
                    'rmse_v':rmse_v,
                    'r2_v':r2_v
                }
            # print('output made')
            metrics.append(output)
            # print('output append')
    return pd.DataFrame(metrics)