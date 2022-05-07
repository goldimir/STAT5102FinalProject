# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:17:27 2022

@author: kingchaucheung
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.api as sm
from scipy import stats
from scipy.stats import wishart, multivariate_normal, t, invgamma, norm, uniform
from scipy.stats import dirichlet, skew, kurtosis,kde, chi2, f, gamma
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
import scipy.interpolate as interpolate
from scipy.interpolate import BSpline, splrep, splev
from itertools import chain, combinations
from scipy.stats import levene
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from scipy.stats import zscore
from numpy import linalg as LA
import stemgraphic
from IPython.display import Image
import STAT5102C2 as C2

#suppress warning
import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 16, 9

def cal_MSC(s,X,Y):
    n = len(Y)
    if X.ndim==1:
        p=1
    else:
        p=X.shape[1]
    XF = np.column_stack((np.ones(n),X))
    X2_inv = np.linalg.inv(XF.T.dot(XF)+0.0000001*np.eye(p+1))
    H = XF.dot(X2_inv)
    H = H.dot(XF.T)
    pY = H.dot(Y)
    mY = np.mean(Y)*np.ones(n)
    e = Y - pY
    me = Y - mY
    SSEF = e.T.dot(e)
    MSEF = SSEF/(n-p-1)
    SST = me.T.dot(me)
    MST = SST/(n-1)
    R2F = 1-SSEF/SST
    AdjR2F = 1-MSEF/MST
    XS = X[:,s]
    XS = np.column_stack((np.ones(n),XS))
    XS2_inv = np.linalg.inv(XS.T.dot(XS)+0.0000001*np.eye(len(s)+1))
    beta = XS2_inv.dot(XS.T)
    HS = XS.dot(beta)
    beta = beta.dot(Y)
    pYS = HS.dot(Y)
    eS = Y - pYS
    SSES = eS.T.dot(eS)
    MSES = SSES/(n-len(s)-1)
    R2 = 1-SSES/SST
    AdjR2 = 1-MSES/MST
    Cp = SSES/MSEF - n + 2*len(s) + 2
    DHS = np.diag(HS)
    eh = [eS[i]/(1-DHS[i]) for i in range(n)]
    PRESS = sum([x*x for x in eh])
    beta_var = np.diag(MSES*XS2_inv)
    t_values = np.abs(beta/np.sqrt(beta_var))
    p_values = 2*(1-t.cdf(t_values,n-len(s)-1))
    output = {}
    output['1-R2'] = 1-R2
    output['1-AdjR2'] = 1-AdjR2
    output['Cp'] = Cp
    output['PRESS'] = np.sqrt(PRESS/n)
    output['AIC'] = n*np.log(SSES/n) + 2*(len(s)+1)
    output['BIC'] = n*np.log(SSES/n) + (len(s)+1)*np.log(n)
    output['Full AdjR2'] = AdjR2F
    output['Full R2'] = R2F
    output['beta'] = beta
    output['p_values'] = p_values
    return output

def gen_combinations(input):
    return sum([list(map(list, combinations(input, i))) for i in range(1,len(input) + 1)], [])

def compare_models(X,Y,pname):
    models = []
    for s in gen_combinations(range(len(pname))):
        fs = cal_MSC(s,X,Y)
        names = [pname[i] for i in s]
        models.append((names,fs['1-R2'],fs['1-AdjR2'], fs['Cp'], fs['PRESS'], fs['AIC'], fs['BIC']))
    df = pd.DataFrame(models)
    df.columns=['model','1-R2','1-AdjR2','Cp','PRESS', 'AIC', 'BIC']
    return df

def cal_p(s,X,Y):
    n = len(Y)
    XS = X[:,s]
    XS = np.column_stack((np.ones(n),XS))
    XS2_inv = np.linalg.inv(XS.T.dot(XS)+0.0000001*np.eye(len(s)+1))
    beta = XS2_inv.dot(XS.T)
    HS = XS.dot(beta)
    beta = beta.dot(Y)
    pYS = HS.dot(Y)
    eS = Y - pYS
    SSES = eS.T.dot(eS)
    MSES = SSES/(n-len(s)-1)
    beta_var = np.diag(MSES*XS2_inv)
    t_values = np.abs(beta/np.sqrt(beta_var))
    p_values = 2*(1-t.cdf(t_values,n-len(s)-1))
    return p_values[1:]

def forward_selection(X,Y,pname,sle=0.05):
    n = len(Y)
    p=X.shape[1]
    Not_selected_list = [x for x in range(p)]
    selected_list = []
    report=[]
    stop = 0
    stage=0
    while stop==0:
        if len(Not_selected_list) > 0:
            pvals = []
            for i in range(len(Not_selected_list)):
                x = Not_selected_list[i]
                s = selected_list+[x]
                pvals.append(cal_p(s,X,Y)[-1])
            if len(pvals) > 0:
                if min(pvals) <= sle:
                    selected_list.append(Not_selected_list[np.argmin(pvals)])
                    stage=stage+1
                    report.append((pname[Not_selected_list[np.argmin(pvals)]],min(pvals),stage))
                    Not_selected_list = [x for x in Not_selected_list if x not in selected_list]
                    if len(Not_selected_list) == 0:
                        stop=1
                else:
                    stop=1
            else:
                stop=1
        else:
            stop = 1
    return report

def backward_elimination(X,Y,pname,sls=0.05):
    n = len(Y)
    p=X.shape[1]
    Not_selected_list = []
    selected_list = [x for x in range(p)]
    report=[]
    stop = 0
    stage=0
    while stop==0:
        if len(selected_list) > 0:
            pvals = cal_p(selected_list,X,Y)
            if max(pvals) >= sls:
                Not_selected_list.append(selected_list[np.argmax(pvals)])
                stage=stage+1
                report.append((pname[selected_list[np.argmax(pvals)]],max(pvals),stage))
                selected_list = [x for x in selected_list if x not in Not_selected_list]
            else:
                stop=1
        else:
            stop=1
    output={}
    output['report'] = report
    output['selected'] = [pname[x] for x in selected_list]
    return output

def stepwise_regression(X,Y,pname,sle=0.05,sls=0.05):
    n = len(Y)
    p=X.shape[1]
    Not_selected_list = [x for x in range(p)]
    selected_list = []
    report=[]
    stop = 0
    while stop==0:
        pvals = []
        if len(Not_selected_list) > 0:
            for i in range(len(Not_selected_list)):
                x = Not_selected_list[i]
                s = selected_list+[x]
                pvals.append(cal_p(s,X,Y)[-1])
            if min(pvals) <= sle:
                selected_list.append(Not_selected_list[np.argmin(pvals)])
                report.append((pname[Not_selected_list[np.argmin(pvals)]],min(pvals),1))
                Not_selected_list = [x for x in Not_selected_list if x not in selected_list]
                pvals = cal_p(selected_list,X,Y)
                if max(pvals) >= sls:
                    Not_selected_list.append(selected_list[np.argmax(pvals)])
                    report.append((pname[selected_list[np.argmax(pvals)]],max(pvals),-1))
                    selected_list = [x for x in selected_list if x not in Not_selected_list]
            else:
                pvals = cal_p(selected_list,X,Y)
                if max(pvals) >= sls:
                    Not_selected_list.append(selected_list[np.argmax(pvals)])
                    report.append((pname[selected_list[np.argmax(pvals)]],max(pvals),-1))
                    selected_list = [x for x in selected_list if x not in Not_selected_list]
                else:
                    stop=1
        else:
            stop=1
    output={}
    output['report'] = report
    output['selected'] = [pname[x] for x in selected_list]
    return output

def reg_analysis(s,X,Y):
    n = len(Y)
    XS = X[:,s]
    XS = np.column_stack((np.ones(n),XS))
    XS2_inv = np.linalg.inv(XS.T.dot(XS)+0.0000001*np.eye(len(s)+1))
    beta = XS2_inv.dot(XS.T)
    HS = XS.dot(beta)
    beta = beta.dot(Y)
    pYS = HS.dot(Y)
    eS = Y - pYS
    SSES = eS.T.dot(eS)
    MSES = SSES/(n-len(s)-1)
    beta_var = np.diag(MSES*XS2_inv)
    t_values = np.abs(beta/np.sqrt(beta_var))
    p_values = 2*(1-t.cdf(t_values,n-len(s)-1))
    output = {}
    output['beta'] = beta
    output['beta_var'] = beta_var
    output['t_values'] = t_values
    output['p_values'] = p_values
    output['SSE'] = SSES
    output['pY'] = pYS
    return output

def stagewise_regression(X,Y,pname,eta=0.05):
    pname = list(pname)
    n = len(Y)
    p=X.shape[1]
    e = Y
    beta = np.zeros(p+1)
    stop = 0
    while stop == 0:
        SSES = []
        models = []
        for j in range(p):
            model = reg_analysis([j],X,e)
            SSES.append(model['SSE'])
            models.append(model)
        m = np.argmin(SSES)
        if models[m]['p_values'][1]>eta:
            stop = 1
        else:
            beta[0] = beta[0] + models[m]['beta'][0]
            beta[m+1] = beta[m+1] + models[m]['beta'][1]
            e = e - models[m]['pY']
    return dict([(x,y) for y,x in zip(beta, ['Intercept']+pname)])

def BP_test(X,Y):
    if np.ndim(X)==1:
        model = C2.cal_multiple_reg(X[:,np.newaxis],Y,eta=0.05)
        n = model['n']
        p = model['p']
        SSE = model['ANOVA']['SSE'][0]
        e = model['residuals']
        g = n*e*e/SSE
        gmodel = C2.cal_multiple_reg(X[:,np.newaxis],g,eta=0.05)
        F = gmodel['ANOVA']['F'][0]
        pval = gmodel['ANOVA']['p_value']
    else:
        model = C2.cal_multiple_reg(X,Y,eta=0.05)
        n = model['n']
        p = model['p']
        SSE = model['ANOVA']['SSE'][0]
        e = model['residuals']
        g = n*e*e/SSE
        gmodel = C2.cal_multiple_reg(X,g,eta=0.05)
        F = gmodel['ANOVA']['F'][0]
        pval = gmodel['ANOVA']['p_value']
    return F,pval

def var_test(A,B):
    nA = len(A)
    nB = len(B)
    vA = np.var(A)
    vB = np.var(B)
    F = vA/vB
    pval = 2*min(f.cdf(F,nA-1,nB-1),1-f.cdf(F,nA-1,nB-1))
    return F,pval

def cal_cook_dist(model):
    p = model['p']
    n = model['n']
    e = model['residuals']
    MSE = model['para']['MSE']
    H = np.diag(model['H'])
    COOKD = []
    for i in range(len(e)):
        a = e[i]*e[i]/((p+1)*MSE)
        b = H[i]/((1-H[i])*(1-H[i]))
        COOKD.append(a*b)
    return np.array(COOKD)

def partial_reg_plot(k,X,Y, pname, labels, u=5):
    p = X.shape[1]
    idx = [i for i in range(p) if i != k]
    X_k = X[:,idx]
    model1 = C2.cal_multiple_reg(X_k,Y)
    Xk = X[:,k]
    model2 = C2.cal_multiple_reg(X_k,Xk)
    eY = model1['residuals']
    eX = model2['residuals']
    m, b = np.polyfit(eX, eY, 1)
    model = C2.cal_multiple_reg(eX,eY)
    err = cal_cook_dist(model)
    lpf = [labels[x] for x in np.argsort(np.abs(err))]
    erf = [eX[x] for x in np.argsort(np.abs(err))]
    spf = [eY[x] for x in np.argsort(np.abs(err))]

    plt.scatter(eX,eY,color='red')
    plt.plot(eX,m*eX+b,color='green')
    plt.xlabel(f'Partial Regression Residuals on {pname[k]}')
    plt.ylabel('Partial Regression Residuals on Y')
    for c in range(1,u+1):
        plt.annotate(lpf[-c], (erf[-c], spf[-c]))
        
    plt.title('Partial Regression Plot')

def partial_plots(k,X,Y, pname, labels, u=5):
    p = X.shape[1]
    idx = [i for i in range(p) if i != k]
    X_k = X[:,idx]
    model1 = C2.cal_multiple_reg(X_k,Y)
    Xk = X[:,k]
    model2 = C2.cal_multiple_reg(X_k,Xk)
    eY = model1['residuals']
    eX = model2['residuals']
    model3 = C2.cal_multiple_reg(eX,eY)
    Z = np.argsort(eX)
    eX2 = [eX[i] for i in Z]
    eY2 = [eY[i] for i in Z]
    pY2 = [model3['pred_Y'][i] for i in Z]
    err1 = cal_cook_dist(model3)
    lpf1 = [labels[x] for x in np.argsort(np.abs(err1))]
    erf1 = [eX[x] for x in np.argsort(np.abs(err1))]
    spf1 = [eY[x] for x in np.argsort(np.abs(err1))]
    
    model4 = C2.cal_multiple_reg(X,Y)
    ZX = np.argsort(Xk)
    Xk2 = [Xk[i] for i in ZX]
    R2 = [model4['residuals'][i]+model4['para']['beta'][k+1]*Xk[i] for i in ZX]
    model5 = C2.cal_multiple_reg(Xk2,R2)
    err2 = cal_cook_dist(model5)
    lpf2 = [labels[x] for x in np.argsort(np.abs(err2))]
    erf2 = [Xk2[x] for x in np.argsort(np.abs(err2))]
    spf2 = [R2[x] for x in np.argsort(np.abs(err2))]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.scatter(eX2,eY2,color='red')
    ax1.plot(eX2,pY2,'k--')
    for c in range(1,u+1):
        ax1.annotate(lpf1[-c], (erf1[-c], spf1[-c]))
    ax1.set_xlabel(f'Partial Regression Residuals on {pname[k]}')
    ax1.set_ylabel('Partial Regression Residuals on Y')
    ax1.set_title('Partial Regression Plot')
    
    ax2.scatter(Xk2,R2,color='red')
    ax2.plot(Xk2,model5['pred_Y'],'k--')
    for c in range(1,u+1):
        ax2.annotate(lpf2[-c], (erf2[-c], spf2[-c]))
    ax2.set_xlabel(f'{pname[k]}')
    ax2.set_ylabel('Adjusted Y')
    ax2.set_title('Partial Residuals Plot')
    
def gen_studentized_deleted_residuals(model):
    p = model['p']
    n = model['n']
    e = model['residuals']
    MSE = model['para']['MSE']
    H = np.diag(model['H'])
    de = []
    for i in range(len(e)):
        MSEi = (1/(n-p-2))*((n-p-1)*MSE - e[i]*e[i]/(1-H[i]))
        de.append(e[i]/np.sqrt(MSEi*(1-H[i])))
    return np.array(de)

def half_qq_plot(data,labels,k=5):
    n = len(data)
    cutpts = [(n+i)/(2*n+1) for i in range(1,n+1)]
    qt = [norm.ppf(x,loc=np.mean(data),scale=np.std(data)) for x in cutpts]
    qt[-1] = max(data)
    
    # custom grid appearance
    s = sorted(data)
    l = [labels[x] for x in np.argsort(data)]
    plt.plot(sorted(data),qt,'ro',markersize=5)
    plt.plot(sorted(data),sorted(data),'--')
    for u in range(1,k+1):
        plt.annotate(l[-u], (s[-u], qt[-u]))

def rplot(Y,X,labels,k=5):
    n = len(Y)
    model = C2.cal_multiple_reg(X,Y)
    Y_hat = model['pred_Y']
    lmpred = model['lmpred_Y']
    Umpred = model['Umpred_Y']
    
    # custom grid appearance
    s = sorted(X)
    l = [labels[x] for x in np.argsort(X)]
    YY = [Y[x] for x in np.argsort(X)]
    YY_hat = [Y_hat[x] for x in np.argsort(X)]
    lmpred_YY = [lmpred[x] for x in np.argsort(X)]
    Umpred_YY = [Umpred[x] for x in np.argsort(X)]
    plt.scatter(sorted(X),YY,color='red')
    plt.plot(sorted(X),YY_hat)
    plt.plot(sorted(X),lmpred_YY,'g--')
    plt.plot(sorted(X),Umpred_YY,'g--')
    for u in range(1,k+1):
        plt.annotate(l[-u], (s[-u], YY[-u]))

def hplot(model,labels,k=5):
    n = model['n']
    p = model['p']
    H = np.diag(model['H'])
    r = gen_studentized_deleted_residuals(model)
    
    # custom grid appearance
    sh = sorted(H)
    lr = [labels[x] for x in np.argsort(np.abs(r))]
    er = [x for x in np.argsort(np.abs(r))]
    sr = [r[x] for x in np.argsort(np.abs(r))]
    lh = [labels[x] for x in np.argsort(H)]
    eh = [x for x in np.argsort(H)]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    ax1.scatter(er,sr,color='red')
    ax1.set_title('Standardized Residuals')
    for u in range(1,k+1):
        ax1.annotate(lr[-u], (er[-u], sr[-u]))
        
    ax2.scatter(eh,sh,color='red')
    ax2.axhline(y = 2*(p+1)/n, color = 'g', linestyle = '--')
    ax2.set_title('The leverage values')
    for u in range(1,k+1):
        ax2.annotate(lh[-u], (eh[-u], sh[-u]))

def cal_influential_measures(model):
    p = model['p']
    n = model['n']
    e = model['residuals']
    MSE = model['para']['MSE']
    SSE = model['ANOVA']['SSE'][0]
    H = np.diag(model['H'])
    COOKD = []
    DFITS = []
    HADI = []
    PF = []
    RF = []
    for i in range(len(e)):
        MSES = (1/(n-p-2))*((n-p-1)*MSE - e[i]*e[i]/(1-H[i]))
        r = e[i]/(MSES*(1-H[i]))
        D = e[i]/np.sqrt(SSE)
        a = e[i]*e[i]/((p+1)*MSE)
        b = H[i]/((1-H[i])*(1-H[i]))
        COOKD.append(a*b)
        DFITS.append(r*np.sqrt(H[i]/(1-H[i])))
        HADI.append(H[i]/(1-H[i]) + ((p+1)/(1-H[i]))*(D*D/(1-D*D)))
        PF.append(H[i]/(1-H[i]))
        RF.append(((p+1)/(1-H[i]))*(D*D/(1-D*D)))
    output={}
    output['COOKD'] = np.array(COOKD)
    output['DFITS'] = np.array(DFITS)
    output['HADI'] = np.array(HADI)
    output['PF'] = PF
    output['RF'] = RF
    return output

def plot_IM(model,labels,k=5):
    IM = cal_influential_measures(model)
    n = model['n']
    p = model['p']
    HADI = IM['HADI']
    DFITS = IM['DFITS']
    COOKD = IM['COOKD']
    PF = IM['PF']
    RF = IM['RF']
    dist = [np.sqrt(a*a+b*b) for a,b in zip(RF,PF)]
    
    # custom grid appearance
    lc = [labels[x] for x in np.argsort(np.abs(COOKD))]
    ec = [x for x in np.argsort(np.abs(COOKD))]
    sc = [COOKD[x] for x in np.argsort(np.abs(COOKD))]
    lh = [labels[x] for x in np.argsort(np.abs(HADI))]
    eh = [x for x in np.argsort(np.abs(HADI))]
    sh = [HADI[x] for x in np.argsort(np.abs(HADI))]
    ld = [labels[x] for x in np.argsort(np.abs(DFITS))]
    ed = [x for x in np.argsort(np.abs(DFITS))]
    sd = [DFITS[x] for x in np.argsort(np.abs(DFITS))]
    lpf = [labels[x] for x in np.argsort(np.abs(PF))]
    erf = [RF[x] for x in np.argsort(np.abs(dist))]
    spf = [PF[x] for x in np.argsort(np.abs(dist))]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    
    ax1.scatter(ec,sc,color='red')
    ax1.axhline(y = f.ppf(0.5,p+1,n-p-1), color = 'g', linestyle = '--')
    ax1.set_title('COOK Distance')
    for u in range(1,k+1):
        ax1.annotate(lc[-u], (ec[-u], sc[-u]))
        
    ax2.scatter(eh,sh,color='red')
    ax2.set_title('HADI')
    for u in range(1,k+1):
        ax2.annotate(lh[-u], (eh[-u], sh[-u]))
        
    ax3.scatter(ed,sd,color='red')
    ax3.axhline(y = 2*np.sqrt((p+1)/(n-p-1)), color = 'g', linestyle = '--')
    ax3.axhline(y = -2*np.sqrt((p+1)/(n-p-1)), color = 'g', linestyle = '--')
    ax3.set_title('DFITS')
    for u in range(1,k+1):
        ax3.annotate(ld[-u], (ed[-u], sd[-u]))
    
    ax4.scatter(erf,spf,color='red')
    ax4.set_title('The Potential Residual Plot')
    for u in range(1,k+1):
        ax4.annotate(lpf[-u], (erf[-u], spf[-u]))

def find_box_cox(Z,Y):
    n=len(Y)
    if np.ndim(Z)==1:
        p=1
    else:
        p=Z.shape[1]
    #p=Z.shape[1]
    lda = np.pad(np.linspace(-10,10,1000),(0,1),'constant')
    max_logL = -100000000000
    Z = np.column_stack((np.ones(n),Z))
    Z2_inv = np.linalg.inv(Z.T.dot(Z)+0.00001*np.eye(p+1))
    beta = Z2_inv.dot(Z.T)
    H = Z.dot(beta)
    for r in lda:
        if r==0:
            Y2 = np.log(Y)
        else:
            Y2 = (np.power(Y,r)-1)/r
        e = Y2 - H.dot(Y2)
        S2 = e.T.dot(e)/n
        logL = -(n/2)*np.log(S2) + (r-1)*sum(np.log(Y))
        if logL > max_logL:
            max_logL = logL
            max_r = r
    if max_r==0:
        Y2 = np.log(Y)
    else:
        Y2 = (np.power(Y,max_r)-1)/max_r
    beta = beta.dot(Y2)
    e = Y2 - H.dot(Y2)
    S2 = e.T.dot(e)/n
    shapiro_test = stats.shapiro(e)
    kstest = stats.kstest((e-np.mean(e))/np.std(e), 'norm')
    output = {}
    output['Y'] = Y2
    output['max_lda'] = max_r
    output['beta'] = beta
    output['e'] = e
    output['S2'] = S2
    output['shapiro_test'] = shapiro_test
    output['kstest'] = kstest
    return output

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

def PCA(df,pname):
    w,v = LA.eig(df[pname].corr())
    S = df[pname].apply(zscore).dot(v)
    S.columns = pname
    return S

def L1(x):
    return sum(np.abs(x))

def L2(x):
    return x.T.dot(x)

def lf(beta,X,Y,alpha):
    n = len(X)
    #X = np.column_stack((np.ones(n),X))
    e = Y-X.dot(beta)
    SSE = e.T.dot(e)
    return 0.5*SSE + alpha*L1(beta) + (1-alpha)*L2(beta)

def regularized_reg(X,Y,a=-1):
    Nrl = 1000000
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
    Fr = X.T.dot(Y)
    for lr in range(Nrl):
        e = Y-X.dot(beta_mat[lr])
        SSE = e.T.dot(e)
        Sval.append(0.5*SSE + alpha[lr]*L1(beta_mat[lr]) + (1-alpha[lr])*L2(beta_mat[lr]))
    mlr = np.argmin(Sval)
    output={}
    output['beta'] = beta_mat[mlr]
    output['Sval'] = Sval[mlr]
    e = Y-X.dot(beta_mat[mlr])
    output['SSE'] = e.T.dot(e)
    output['alpha_min'] = alpha[mlr]
    return output

def residual_plot(model,dd=5):
    res = gen_studentized_deleted_residuals(model)
    ares = np.abs(res)
    pY = model['pred_Y']
    labels = list(range(len(ares)))
    lpf = [labels[x] for x in np.argsort(ares)]
    erf = [pY[x] for x in np.argsort(ares)]
    spf = [res[x] for x in np.argsort(ares)]
    plt.scatter(pY,res,color='darkblue')
    for c in range(1,dd+1):
        plt.annotate(lpf[-c], (erf[-c], spf[-c]))
    plt.axhline(y=0.0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Standardized residuals')
    plt.title('Standardized residuals versus Predicted values')
