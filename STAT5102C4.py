# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:06:37 2022

@author: kingchaucheung
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import yfinance as yf  
import pandas as pd
import quandl
import numpy as np
import scipy.stats
import statsmodels.api as sm
from scipy import stats
from scipy.stats import wishart, multivariate_normal, t, invgamma, norm, uniform, dirichlet, skew, kurtosis,kde, chi2, f
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import scatter_matrix
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from scipy import linalg
from statsmodels.regression import linear_model
from statsmodels.stats.anova import anova_lm
from numpy import linalg as LA
import stemgraphic
from numpy.random import random
from scipy.stats import rayleigh
import random
from scipy.stats import zscore
from mpl_toolkits.mplot3d.axes3d import get_test_data
from IPython.display import Image
import STAT5102C3 as C3
import STAT5102C2 as C2

import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 16, 9

def autocorr_test(e):
    e1 = e[1:]
    e2 = e[:-1]
    model = C2.cal_multiple_reg(e2,e1,fit_intercept=False)
    return model['para']['p_values']

def cal_VIF(X,pname):
    if np.ndim(X)==1:
        return 'only a single predictor.'
    else:
        p = X.shape[1]
        n = X.shape[0]
        VIF = {}
        for r in range(p):
            idx = [i for i in range(p) if i != r]
            Z = X[:,idx]
            Y = X[:,r]
            if np.ndim(Z)==1:
                Z = Z[:,np.newaxis]
            model = C2.cal_multiple_reg(Z,Y,eta=0.05)
            if (model['ANOVA']['R_squared'] > 0) & (model['ANOVA']['R_squared'] < 1):
                VIF[pname[r]] = (1/(1-model['ANOVA']['R_squared']))
            else:
                VIF[pname[r]] = 'undefined'
        return VIF

#model1 is the reduced model
def compare(model1,model2):
    df1 = model1['ANOVA']['SSE'][1]
    df2 = model2['ANOVA']['SSE'][1]
    SSE1 = model1['ANOVA']['SSE'][0]
    SSE2 = model2['ANOVA']['SSE'][0]
    F = ((SSE1-SSE2)/(df1-df2))/(SSE2/df2)
    return 1-f.cdf(F,df1-df2,df2)

def cal_cond_indices(df,pname):
    v, vc = np.linalg.eig(df[pname].corr())
    z = np.argsort(v)
    vm = v[z[-1]]
    output={}
    output['Kappa'] = pd.DataFrame({'col':pname,'eigenval':v,'cond_index':np.sqrt(vm/v)})
    output['eigenvectors'] = pd.DataFrame(vc,columns=pname)
    return output

def cal_ridge_beta(k,X,Y):
    n=len(Y)
    if np.ndim(X)==1:
        p=1
    else:
        p=X.shape[1]
    X = np.column_stack((np.ones(n),X))
    X2_inv = np.linalg.inv(X.T.dot(X)+k*np.eye(p+1))
    beta = X2_inv.dot(X.T.dot(Y))
    return beta

def cal_ridge_beta2(k,X,Y):
    model = C2.cal_multiple_reg(X,Y)
    MSE = model['para']['MSE']
    n=len(Y)
    if np.ndim(X)==1:
        p=1
    else:
        p=X.shape[1]
    X = np.column_stack((np.ones(n),X))
    X2_inv = np.linalg.inv(X.T.dot(X)+k*np.eye(p+1))
    beta = X2_inv.dot(X.T.dot(Y))
    beta_var = MSE*X2_inv.dot(X.T.dot(X)).dot(X2_inv)
    output={}
    output['beta'] = beta
    output['beta_se'] = np.diag(beta_var)
    return output

def combs(a,b):
    c = []
    for i in a:
        for j in b:
            c.append((i,j))
    return c

#apply transformation
def tV(lda,Y):
    if lda==0:
        Y1 = np.log(Y)
        flag = 'log_tran'
    else:
        Y1 = []
        for x in Y:
            if lda > 0:
                Y1.append(np.power(x,lda))
            else:
                Y1.append(1/np.power(x,-lda))
        Y1 = np.array(Y1)
        flag = f'lda={lda}'
    output = {}
    output['Y'] = Y1
    output['flag'] = flag
    return output

def eval_LS(rho,X,Y):
    DY = Y[1:] - rho*Y[:-1]
    DX = X[1:] - rho*X[:-1]
    if np.abs(rho-1)<0.01:
        model = C2.cal_multiple_reg(DX,DY,fit_intercept=False)
    else:
        model = C2.cal_multiple_reg(DX,DY)
    return model['para']['MSE']

def determine_k(X,Y):
    model = C2.cal_multiple_reg(X,Y)
    MSE = model['para']['MSE']
    beta = model['para']['beta']
    n=len(Y)
    if np.ndim(X)==1:
        p=1
    else:
        p=X.shape[1]
    k=[]
    x = p*MSE/sum([x*x for x in beta])
    if x < 1:
        k.append(p*MSE/sum([x*x for x in beta]))
    else:
        k.append(0)
    stop = 0
    nit = 0
    while stop==0:
        beta = cal_ridge_beta(k[-1],X,Y)
        x = p*MSE/sum([x*x for x in beta])
        if x < 1:
            k.append(p*MSE/sum([x*x for x in beta]))
        else:
            k.append(0)
        if np.abs(k[-1]-k[-2])<0.00000000001:
            stop=1
        else:
            nit = nit+1
            if nit > 1000:
                stop=1
    return k

def cal_BIC(X,Y,lda):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    X = np.column_stack((np.ones(n),X))
    Temp = X.T.dot(X) + lda*np.eye(p+1)
    Temp_inv = np.linalg.inv(Temp)
    H = X.dot(Temp_inv.dot(X.T))
    SSE = Y.T.dot(np.eye(n)-H)
    SSE = SSE.dot(Y)
    logL = n*np.log(SSE/n) + n
    return 2*np.log(n)*np.trace(H) + logL

def cal_PRESS(X,Y,lda):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    X = np.column_stack((np.ones(n),X))
    err = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Xi = X[idx]
        Yi = Y[idx]
        Temp = Xi.T.dot(Xi) + lda*np.eye(p+1)
        Temp_inv = np.linalg.inv(Temp)
        beta = Temp_inv.dot(Xi.T.dot(Yi))
        pYi = X[i].T.dot(beta)
        err.append(Y[i]-pYi)
    err2 = [x*x for x in err]
    return np.sqrt(sum(err2)/n)

def cal_GCV(X,Y,lda):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    X = np.column_stack((np.ones(n),X))
    Temp = X.T.dot(X) + lda*np.eye(p+1)
    Temp_inv = np.linalg.inv(Temp)
    H = X.dot(Temp_inv.dot(X.T))
    k = np.trace(H)
    err = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Xi = X[idx]
        Yi = Y[idx]
        Temp = Xi.T.dot(Xi) + lda*np.eye(p+1)
        Temp_inv = np.linalg.inv(Temp)
        beta = Temp_inv.dot(Xi.T.dot(Yi))
        pYi = X[i].T.dot(beta)
        err.append((Y[i]-pYi)/(1-(k/n)))
    err2 = [x*x for x in err]
    return np.sqrt(sum(err2)/n)

def find_lasso_beta(X,Y,lda,gamma=0.01,tol=0.1):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    X = np.column_stack((np.ones(n),X))
    Temp = X.T.dot(X) + lda*np.eye(p+1)
    Temp_inv = np.linalg.inv(Temp)
    beta = Temp_inv.dot(X.T.dot(Y))
    stop = 0
    it = 0
    while stop==0:
        it = it+1
        beta0 = beta
        DB = -X.T.dot(Y) + X.T.dot(X.dot(beta)) + lda*np.sign(beta)
        beta = beta - gamma*DB
        if sum(np.abs(beta - beta0)) < tol:
            stop = 1
        elif it > 1000:
            stop = 1
    output = {}
    output['beta'] = beta
    e = Y - X.dot(beta)
    output['SSE'] = np.sqrt((1/(n-p-1))*e.T.dot(e))
    output['e'] = e
    return output

def cal_PRESS_lasso(X,Y,lda):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    XX = np.column_stack((np.ones(n),X))
    err = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Xi = X[idx]
        Yi = Y[idx]
        model = find_lasso_beta(Xi,Yi,lda)
        beta = model['beta']
        pYi = XX[i].T.dot(beta)
        err.append(Y[i]-pYi)
    err2 = [x*x for x in err]
    return np.sqrt(sum(err2)/n)

def L1(x):
    return sum(np.abs(x))

def L2(x):
    return np.sqrt(x.T.dot(x))

def regularized_reg(X,Y,a=-1,lda=1):
    Nrl = 100000
    n=len(Y)
    if np.ndim(X) == 1:
        X = X[:,np.newaxis]
    p=X.shape[1]
    X = np.column_stack((np.ones(n),X))
    X2_inv = np.linalg.inv(X.T.dot(X))
    beta = X2_inv.dot(X.T.dot(Y))
    H = X.dot(X2_inv).dot(X.T)
    pred_Y = H.dot(Y)
    e = Y - pred_Y
    mc = Y - np.mean(Y)*np.ones(n)
    SST = mc.T.dot(mc)
    SSE = e.T.dot(e)
    SSR = SST - SSE
    MSE = SSE/(n-p-1)
    MSR = SSR/p
    F = MSR/MSE
    beta_var = MSE*X2_inv
    beta_se = np.sqrt(np.diag(beta_var))
    if a==-1:
        alpha = uniform.rvs(size=Nrl)
    elif np.ndim(a) == 0:
        alpha = a*np.ones(Nrl)
    else:
        alpha = a
    mean_beta = beta
    beta_mat = np.random.multivariate_normal(mean_beta, beta_var, Nrl)
    Sval = []
    #Sval_min = 1000000000000000000
    Fr = X.T.dot(Y)
    for lr in range(Nrl):
        #beta_mat = np.random.multivariate_normal(mean_beta, beta_var, size=1)[0]
        e = Y-X.dot(beta_mat[lr])
        SSE = e.T.dot(e)
        L = 0.5*np.sqrt(SSE) + lda*(alpha[lr]*L1(beta_mat[lr]) + (1-alpha[lr])*L2(beta_mat[lr]))
        Sval.append(L)
    mlr = np.argmin(Sval)
    output={}
    output['beta'] = beta_mat[mlr]
    output['Sval'] = Sval[mlr]
    e = Y-X.dot(beta_mat[mlr])
    output['SSE'] = e.T.dot(e)
    output['alpha_min'] = alpha[mlr]
    output['alpha'] = alpha
    return output

def gen_interactions(X,de,pname):
    s = C3.gen_combinations(range(len(pname)))
    VX = []
    VarX = []
    for i in range(len(s)):
        if len(s[i]) <= de:
            VX.append(np.prod(X[:,s[i]],axis=1))
            VarX.append(tuple([pname[x] for x in s[i]]))
    return np.array(VX), VarX

def wls(X,Y,W):
    n = len(Y)
    if np.ndim(X)==1:
        p=1
    else:
        p=X.shape[1]
    X = np.column_stack((np.ones(n),X))
    XS = sqrtm(W).dot(X)
    YS = sqrtm(W).dot(Y)
    return C2.cal_multiple_reg(XS,YS,fit_intercept=False)