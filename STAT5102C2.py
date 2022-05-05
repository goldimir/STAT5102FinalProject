# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from numpy.random import random
from scipy.stats import rayleigh
from scipy.stats import zscore
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.axes3d import get_test_data
import itertools

#suppress warning
import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 16, 9

def cal_multiple_reg(X,Y,eta=0.05,fit_intercept=True):
    n=len(Y)
    if np.ndim(X)==1:
        p=1
    else:
        p=X.shape[1]
    if fit_intercept==True:
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
        MST = SST/(n-1)
        F = MSR/MSE
        beta_var = MSE*X2_inv
        beta_se = np.sqrt(np.diag(beta_var))
        beta_T = beta/beta_se
        R2 = SSR/SST
        AR2 = (1-R2)*(n-1)/(n-p-1)
        AR2 = 1-AR2
        lmpred_Y = []
        Umpred_Y = []
        for i in range(len(Y)):
            x = X[i]
            y = x.T.dot(beta)
            y_var = MSE*x.T.dot(X2_inv).dot(x)
            lmpred_Y.append(y - t.ppf(1-eta/2,n-p-1)*np.sqrt(y_var))
            Umpred_Y.append(y + t.ppf(1-eta/2,n-p-1)*np.sqrt(y_var))
        output={}
        output['n'] = n
        output['p'] = p
        output['ANOVA'] = {}
        output['ANOVA']['SSE'] = (SSE,n-p-1,MSE)
        output['ANOVA']['SSR'] = (SSR,p,MSR)
        output['ANOVA']['SST'] = (SST,n-1,MST)
        output['ANOVA']['F'] = (F,p,n-p-1)
        output['ANOVA']['p_value'] = 1-f.cdf(F,p,n-p-1)
        output['ANOVA']['R_squared'] = R2
        output['ANOVA']['Adjusted R_squared'] = AR2
        output['para']={}
        output['para']['beta'] = beta
        output['para']['beta_cov'] = beta_var
        output['para']['beta_se'] = beta_se
        output['para']['beta_T'] = beta_T
        output['para']['MSE'] = MSE
        output['para']['p_values'] = 2*(1-t.cdf(np.abs(beta_T),n-p-1))
        output['pred_Y'] = pred_Y
        output['lmpred_Y'] = lmpred_Y
        output['Umpred_Y'] = Umpred_Y
        output['interval'] = {}
        output['interval']['beta'] = (beta - t.ppf(1-eta/2,n-p-1)*beta_se, beta + t.ppf(1-eta/2,n-p-1)*beta_se)
        output['interval']['sigma2'] = ((n-p-1)*MSE/chi2.ppf(1-eta/2,n-p-1),(n-p-1)*MSE/chi2.ppf(eta/2,n-p-1))
        output['p_values'] = {}
        output['p_values']['beta'] = 2*(1-t.cdf(np.abs(beta_T),n-p-1))
        output['p_values']['beta_T'] = beta_T
        output['residuals'] = e
        output['H'] = H
        output['X2_inv'] = X2_inv
        output['ANOVA']['negloglike'] = (n/2)*np.log(MSE)
        output['fit_intercept'] = True
    else:
        if p > 1:
            X2_inv = np.linalg.inv(X.T.dot(X))
            beta = X2_inv.dot(X.T.dot(Y))
            H = X.dot(X2_inv).dot(X.T)
            pred_Y = H.dot(Y)
            e = Y - pred_Y
            #mc = Y - np.mean(Y)*np.ones(n)
            SST = Y.T.dot(Y)
            SSE = e.T.dot(e)
            SSR = SST - SSE
            MSE = SSE/(n-p)
            MSR = SSR/p
            MST = SST/n
            F = MSR/MSE
            beta_var = MSE*X2_inv
            beta_se = np.sqrt(np.diag(beta_var))
            beta_T = beta/beta_se
            R2 = SSR/SST
            AR2 = (1-R2)*n/(n-p)
            AR2 = 1-AR2
            lmpred_Y = []
            Umpred_Y = []
            lpred_Y = []
            Upred_Y = []
            for i in range(len(Y)):
                x = X[i]
                y = x.T.dot(beta)
                y_var = MSE*x.T.dot(X2_inv).dot(x)
                y_var2 = MSE*(1+x.T.dot(X2_inv).dot(x))
                lmpred_Y.append(y - t.ppf(1-eta/2,n-p)*np.sqrt(y_var))
                Umpred_Y.append(y + t.ppf(1-eta/2,n-p)*np.sqrt(y_var))
                lpred_Y.append(y - t.ppf(1-eta/2,n-p)*np.sqrt(y_var2))
                Upred_Y.append(y + t.ppf(1-eta/2,n-p)*np.sqrt(y_var2))
            output={}
            output['n'] = n
            output['p'] = p
            output['ANOVA'] = {}
            output['ANOVA']['SSE'] = (SSE,n-p,MSE)
            output['ANOVA']['SSR'] = (SSR,p,MSR)
            output['ANOVA']['SST'] = (SST,n,MST)
            output['ANOVA']['F'] = (F,p,n-p)
            output['ANOVA']['p_value'] = 1-f.cdf(F,p,n-p)
            output['ANOVA']['R_squared'] = R2
            output['ANOVA']['Adjusted R_squared'] = AR2
            output['ANOVA']['negloglike'] = (n/2)*np.log(MSE)
            output['para']={}
            output['para']['beta'] = beta
            output['para']['beta_cov'] = beta_var
            output['para']['beta_se'] = beta_se
            output['para']['beta_T'] = beta_T
            output['para']['MSE'] = MSE
            output['para']['p_values'] = 2*(1-t.cdf(np.abs(beta_T),n-p))
            output['pred_Y'] = pred_Y
            output['lmpred_Y'] = lmpred_Y
            output['Umpred_Y'] = Umpred_Y
            output['lpred_Y'] = lpred_Y
            output['Upred_Y'] = Upred_Y
            output['interval'] = {}
            output['interval']['beta'] = (beta - t.ppf(1-eta/2,n-p)*beta_se, beta + t.ppf(1-eta/2,n-p)*beta_se)
            output['interval']['sigma2'] = ((n-p)*MSE/chi2.ppf(1-eta/2,n-p),(n-p)*MSE/chi2.ppf(eta/2,n-p))
            output['p_values'] = {}
            output['p_values']['beta'] = 2*(1-t.cdf(np.abs(beta_T),n-p))
            output['p_values']['beta_T'] = beta_T
            output['residuals'] = e
            output['H'] = H
            output['X2_inv'] = X2_inv
            output['fit_intercept'] = False
        else:
            X2_inv = 1/(X.T.dot(X))
            beta = X2_inv*(X.T.dot(Y))
            H = (X2_inv)*np.outer(X,X)
            pred_Y = H.dot(Y)
            e = Y - pred_Y
            #mc = Y - np.mean(Y)*np.ones(n)
            SST = Y.T.dot(Y)
            SSE = e.T.dot(e)
            SSR = SST - SSE
            MSE = SSE/(n-p)
            MSR = SSR/p
            MST = SST/n
            F = MSR/MSE
            beta_var = MSE*X2_inv
            beta_se = np.sqrt(beta_var)
            beta_T = beta/beta_se
            R2 = SSR/SST
            AR2 = (1-R2)*n/(n-p)
            AR2 = 1-AR2
            lmpred_Y = []
            Umpred_Y = []
            for i in range(len(Y)):
                x = X[i]
                y = x*beta
                y_var = MSE*X2_inv*x*x
                lmpred_Y.append(y - t.ppf(1-eta/2,n-p)*np.sqrt(y_var))
                Umpred_Y.append(y + t.ppf(1-eta/2,n-p)*np.sqrt(y_var))
            output={}
            output['n'] = n
            output['p'] = p
            output['ANOVA'] = {}
            output['ANOVA']['SSE'] = (SSE,n-p,MSE)
            output['ANOVA']['SSR'] = (SSR,p,MSR)
            output['ANOVA']['SST'] = (SST,n,MST)
            output['ANOVA']['F'] = (F,p,n-p)
            output['ANOVA']['p_value'] = 1-f.cdf(F,p,n-p)
            output['ANOVA']['R_squared'] = R2
            output['ANOVA']['Adjusted R_squared'] = AR2
            output['ANOVA']['negloglike'] = (n/2)*np.log(MSE)
            output['para']={}
            output['para']['beta'] = beta
            output['para']['beta_cov'] = beta_var
            output['para']['beta_se'] = beta_se
            output['para']['beta_T'] = beta_T
            output['para']['MSE'] = MSE
            output['para']['p_values'] = 2*(1-t.cdf(np.abs(beta_T),n-p))
            output['pred_Y'] = pred_Y
            output['lmpred_Y'] = lmpred_Y
            output['Umpred_Y'] = Umpred_Y
            output['interval'] = {}
            output['interval']['beta'] = (beta - t.ppf(1-eta/2,n-p)*beta_se, beta + t.ppf(1-eta/2,n-p)*beta_se)
            output['interval']['sigma2'] = ((n-p)*MSE/chi2.ppf(1-eta/2,n-p),(n-p)*MSE/chi2.ppf(eta/2,n-p))
            output['p_values'] = {}
            output['p_values']['beta'] = 2*(1-t.cdf(np.abs(beta_T),n-p))
            output['p_values']['beta_T'] = beta_T
            output['residuals'] = e
            output['H'] = H
            output['X2_inv'] = X2_inv
            output['fit_intercept'] = False
    return output

def print_multi_reg(model,var_name,dep_name):
    fi = model['fit_intercept']
    X = var_name[0]
    Y = dep_name
    n = model['n']
    R2 = model['ANOVA']['R_squared']
    AR2 = model['ANOVA']['Adjusted R_squared']
    logL = model['ANOVA']['negloglike']
    SSR = model['ANOVA']['SSR'][0]
    SSR_df = model['ANOVA']['SSR'][1]
    MSR = model['ANOVA']['SSR'][2]
    SSE = model['ANOVA']['SSE'][0]
    SSE_df = model['ANOVA']['SSE'][1]
    MSE = model['ANOVA']['SSE'][2]
    SST = model['ANOVA']['SST'][0]
    SST_df = model['ANOVA']['SST'][1]
    MST = model['ANOVA']['SST'][2]
    F_stat = model['ANOVA']['F'][0]
    p_val = model['ANOVA']['p_value']
    if p_val < 0.0001:
        p_val = '<0.0001'
    else:
        p_val = round(p_val,4)
    beta = model['para']['beta']
    beta_se = model['para']['beta_se']
    beta_T = model['p_values']['beta_T']
    beta_pval = model['p_values']['beta']
    bpval = []
    for pv in beta_pval:
        if pv < 0.0001:
            bpval.append('<0.0001')
        else:
            bpval.append(round(pv,4))
    print(f'\t\t\t\t     Multiple Regression Result')
    print('=================================================================================================')
    print(f'Dependent variable:\t\t\t{Y}\tNegative log-likelihood:\t\t{round(logL,4)}')
    print(f'Model:\t\t\t   Multiple Regression\tNo. of observations\t\t\t{n}')
    print(f'R-squared:\t\t\t        {round(R2,4)}\tAdjusted R-squared\t\t\t{round(AR2,4)}')
    print('=================================================================================================')
    print(f'\t\tSum of Squares (SS)\tdf\tMean Squares (MS)\tF values\tp-value')
    print('=================================================================================================')
    print(f'Regression\t{round(SSR,4):16.4f}\t{SSR_df}\t{round(MSR,4):16.4f}\t{round(F_stat,4):9.4f}\t{p_val}')
    print(f'Error\t\t{round(SSE,4):16.4f}\t{SSE_df}\t{round(MSE,4):16.4f}')
    print(f'Total\t\t{round(SST,4):16.4f}\t{SST_df}\t{round(MST,4):16.4f}')
    print('=================================================================================================')
    print('\t\t\t\t     Parameter Estimates')
    print('=================================================================================================')
    print(f'Variable\tLabel\t\tdf\tParameter\tStandard\tt Value\t\tp value')
    print(f'\t\t\t\t\tEstimate\tError')
    print('=================================================================================================')
    if fi==True:
        print(f'Intercept\tIntercept\t1\t{round(beta[0],6):10.6f}\t{round(beta_se[0],6):10.6f}\t{round(beta_T[0],6):10.6f}\t{bpval[0]}')
        for i in range(len(var_name)):
            print(f'X{i+1}\t\t{var_name[i]}\t\t1\t{round(beta[i+1],6):10.6f}\t{round(beta_se[i+1],6):10.6f}\t{round(beta_T[i+1],6):10.6f}\t{bpval[i+1]}')
    else:
        for i in range(len(var_name)):
            print(f'X{i+1}\t\t{var_name[i]}\t\t1\t{round(beta[i],6):10.6f}\t{round(beta_se[i],6):10.6f}\t{round(beta_T[i],6):10.6f}\t{bpval[i]}')
    print('=================================================================================================')

#model1 is the reduced model
#model2 is the full model
def compare(model1,model2):
    df1 = model1['ANOVA']['SSE'][1]
    df2 = model2['ANOVA']['SSE'][1]
    SSE1 = model1['ANOVA']['SSE'][0]
    SSE2 = model2['ANOVA']['SSE'][0]
    F = ((SSE1-SSE2)/(df1-df2))/(SSE2/df2)
    return pd.DataFrame({'F-Stat':[F],'df1':[df1-df2],'df2':[df2],'p-value':[1-f.cdf(F,df1-df2,df2)]})

def cal_predictions(reg_model,X_new,eta=0.05):
    n = reg_model['n']
    p = reg_model['p']
    H = reg_model['H']
    beta = reg_model['para']['beta']
    MSE = reg_model['para']['MSE']
    X2_inv = reg_model['X2_inv']
    ne = np.ndim(X_new)
    pred_Y_new = []
    lpred_Y_new = []
    Upred_Y_new = []
    lmpred_Y_new = []
    Umpred_Y_new = []
    if ne > 1:
        X_new = np.column_stack((np.ones(len(X_new)),X_new))
        for i in range(len(X_new)):
            x = X_new[i]
            y = x.T.dot(beta)
            y_var = x.T.dot(X2_inv.dot(x))
            pred_Y_new.append(y)
            lmpred_Y_new.append(y - t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*y_var))
            Umpred_Y_new.append(y + t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*y_var))
            lpred_Y_new.append(y - t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*(1+y_var)))
            Upred_Y_new.append(y + t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*(1+y_var)))
    else:
        x = np.array([1]+list(X_new))
        y = x.T.dot(beta)
        y_var = x.T.dot(X2_inv.dot(x))
        pred_Y_new.append(y)
        lmpred_Y_new.append(y - t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*y_var))
        Umpred_Y_new.append(y + t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*y_var))
        lpred_Y_new.append(y - t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*(1+y_var)))
        Upred_Y_new.append(y + t.ppf(1-eta/2,n-p-1)*np.sqrt(MSE*(1+y_var)))
    output={}
    output['pred_Y'] = np.array(pred_Y_new)
    output['lpred_Y'] = np.array(lpred_Y_new)
    output['Upred_Y'] = np.array(Upred_Y_new)
    output['lmpred_Y'] = np.array(lmpred_Y_new)
    output['Umpred_Y'] = np.array(Umpred_Y_new)
    return pd.DataFrame(output)

def cal_standardized_reg(X,Y,eta=0.05):
    sY = (Y - np.mean(Y))/np.std(Y)
    sX = X
    for i in range(X.shape[1]):
        sX[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
    SM = cal_multiple_reg(sX,sY,eta=eta,fit_intercept=False)
    return SM

def qq_plot(data,bins=25):
    n = len(data)
    cutpts = [1-np.power(0.5,1/n)]+[(i-0.3175)/(n+0.365) for i in range(2,n)]+[np.power(0.5,1/n)]
    qt = [norm.ppf(x,loc=np.mean(data),scale=np.std(data)) for x in cutpts]
    qt[-1] = max(data)
    
    mu, std = norm.fit(data) 
    
    fig, axes = plt.subplots(1, 2, figsize=(18,8))

    # default grid appearance
    axes[0].hist(data, bins=bins, density=True, alpha=0.6, color='deepskyblue',edgecolor='green', linewidth=1)
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
  
    axes[0].plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    axes[0].set_title(title)

    # custom grid appearance
    axes[1].plot(sorted(data),qt,'ro',markersize=2)
    axes[1].plot(sorted(data),sorted(data),'--')

def plot_scatter(X,Y,d,dd,varlabels):
    m = np.polyfit(X, Y, d)
    for i in range(len(m)):
        if i==0:
            predicted = m[-1]
            Z = np.ones(len(X))
        else:
            Z = Z*X
            predicted = predicted + m[-1-i]*Z
    e = np.abs(Y - predicted)
    labels = list(range(len(Y)))
    lpf = [labels[x] for x in np.argsort(e)]
    erf = [X[x] for x in np.argsort(e)]
    spf = [Y[x] for x in np.argsort(e)]
    ls = np.argsort(X)
    XS = [X[s] for s in ls]
    pS = [predicted[s] for s in ls]
    plt.scatter(X,Y, color='purple')
    plt.plot(XS,pS,'r--')
    for c in range(1,dd):
        plt.annotate(lpf[-c], (erf[-c], spf[-c]))
    plt.xlabel(varlabels[0])
    plt.ylabel(varlabels[1])
    plt.title(f'Plot of {varlabels[0]} vs. {varlabels[1]}')

def ceiling(k,m):
    x = divmod(k,m)
    if x[1]>0:
        d = 1
    else:
        d = 0
    return x[0]+d

def plot_mscatter(X,Y,d,dd,xlabels,ylabel):
    h = X.shape[1]
    g = ceiling(h,2)
    plt.figure(figsize=(16, 4*g))
    palette = plt.get_cmap('Set1')
    for k in range(h):
        plt.subplot(g,2,k+1)
        m = np.polyfit(X[:,k], Y, d[k])
        for i in range(len(m)):
            if i==0:
                predicted = m[-1]
                Z = np.ones(len(X[:,k]))
            else:
                Z = Z*X[:,k]
                predicted = predicted + m[-1-i]*Z
        e = np.abs(Y - predicted)
        labels = list(range(len(Y)))
        lpf = [labels[x] for x in np.argsort(e)]
        erf = [X[x,k] for x in np.argsort(e)]
        spf = [Y[x] for x in np.argsort(e)]
        ls = np.argsort(X[:,k])
        XS = [X[s,k] for s in ls]
        pS = [predicted[s] for s in ls]
        plt.scatter(X[:,k],Y, color=palette(k+1))
        plt.plot(XS,pS,'r--')
        for c in range(1,dd):
            plt.annotate(lpf[-c], (erf[-c], spf[-c]))
        plt.xlabel(xlabels[k])
        plt.ylabel(ylabel)
        #plt.title(f'Plot of {xlabels[k]} vs. {ylabel}')  
    #plt.legend(loc='best')
    

def gen_studentized_residuals(model):
    e = model['residuals']
    MSE = model['para']['MSE']
    H = np.diag(model['H'])
    return e/np.sqrt(MSE*(1-H))

def print_conf_interval(model,var_name):
    fi = model['fit_intercept']
    LL = model['interval']['beta'][0]
    UL = model['interval']['beta'][1]
    MSE_int = model['interval']['sigma2']
    beta = model['para']['beta']
    beta_se = model['para']['beta_se']
    print('===============================================================================================')
    print(f'Variable\tEstimate\t\tS.E.\t\t95% Lower Limit\t\t95% Upper Limit')
    print('===============================================================================================')
    if fi==True:
        print(f'Intercept\t{round(beta[0],6):10.6f}\t{round(beta_se[0],4):10.4f}\t\t{round(LL[0],6):10.6f}\t\t{round(UL[0],6):10.6f}')
        for i in range(len(var_name)):
            print(f'{var_name[i]}\t\t{round(beta[i+1],6):10.6f}\t{round(beta_se[i+1],4):10.4f}\t\t{round(LL[i+1],6):10.6f}\t\t{round(UL[i+1],6):10.6f}')
    else:
        for i in range(len(var_name)):
            print(f'{var_name[i]}\t\t{round(beta[i],6):10.6f}\t{round(beta_se[i],4):10.4f}\t\t{round(LL[i],6):10.6f}\t\t{round(UL[i],6):10.6f}')

def cal_confint_of_beta(ve,model,eta=0.05):
    beta_cov = model['para']['beta_cov']
    beta = model['para']['beta']
    MSE = model['para']['MSE']
    est = ve.T.dot(beta)
    est_var = ve.T.dot(beta_cov.dot(ve))
    df = model['n'] - model['p'] - 1
    output = {}
    output['estimate'] = est
    output['se'] = np.sqrt(est_var)
    output['lower'] = est - t.ppf(1-eta/2,df)*np.sqrt(est_var)
    output['upper'] = est + t.ppf(1-eta/2,df)*np.sqrt(est_var)
    return output

#train-test split
def train_test_split(data,r=0.7):
    arr = np.arange(len(data))
    np.random.shuffle(arr)
    m=np.floor(len(data)*r).astype('int')
    train_idx = list(arr[:m])
    test_idx = list(arr[m:])
    return data.iloc[train_idx], data.iloc[test_idx], train_idx, test_idx

def cal_SScontribution(i,X,Y):
    n=len(Y)
    p=X.shape[1]
    X_s = np.column_stack((np.ones(n),X))
    X2_inv = np.linalg.inv(X_s.T.dot(X_s))
    H = X_s.dot(X2_inv).dot(X_s.T)
    pred_Y = H.dot(Y)
    e = Y - pred_Y
    SSE_full = e.T.dot(e)
    X1 = np.delete(X, i, axis=1)
    X1_s = np.column_stack((np.ones(n),X1))
    X12_inv = np.linalg.inv(X1_s.T.dot(X1_s))
    H1 = X1_s.dot(X12_inv).dot(X1_s.T)
    pred_Y1 = H1.dot(Y)
    e1 = Y - pred_Y1
    SSE = e1.T.dot(e1)
    output={}
    output['SSE_full'] = SSE_full
    output['SSE'] = SSE
    output['SS_Contribution'] = SSE - SSE_full
    return output

def generate_Type_I_SS(df,var_name,dep):
    X = df[var_name].to_numpy()
    Y = df[dep].to_numpy()
    SS = []
    if np.ndim(X)==1:
        model = cal_multiple_reg(X,Y)
        SS.append(model['ANOVA']['SSR'][0])
    else:
        X1 = X[:,0]
        model = cal_multiple_reg(X1,Y)
        SS.append(model['ANOVA']['SSR'][0])
        for i in range(1,X.shape[1]):
            SSRP = model['ANOVA']['SSR'][0]
            X1 = X[:,:(i+1)]
            model = cal_multiple_reg(X1,Y)
            SS.append(model['ANOVA']['SSR'][0]-SSRP)
    model = cal_multiple_reg(X,Y)
    SS = [model['ANOVA']['SST'][0]] + SS
    return tuple(SS)

def gen_all_type_I_SS(df,var_name,dep):
    S = list(itertools.permutations(var_name))
    SSRS = []
    for s in S:
        s = list(s)
        SS = generate_Type_I_SS(df,s,dep)
        SSRS.append((tuple(s),*SS))
    df2 = pd.DataFrame(SSRS)
    return df2

def find_optimal_set_by_SS(df):
    idx_score = []
    for i in range(len(df)):
        Y = 0
        for j in range(3,df.shape[1]):
            if df.iloc[i][j] <= df.iloc[i][j-1]:
                Y = Y+1
        idx_score.append((i,Y))
    idx_score.sort(key=lambda x: x[1],reverse=True)
    y = max([t for _,t in idx_score])
    return df.iloc[[x for x,t in idx_score if t==y]]