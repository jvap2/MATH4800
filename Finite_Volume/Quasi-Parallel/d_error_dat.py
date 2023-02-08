import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


columns=[]
info={}
with open('error.txt') as f:
    for (i,lines) in enumerate(f.readlines()):
        if i==0:
            columns = [item.strip() for item in lines.split(' ')]
            df=pd.DataFrame()
        else:
            data = [item.strip() for item in lines.split(' ')]
            for d,c in zip(data,columns):
                df.loc[i,c]=d

h=np.log10(df['h'].to_numpy(dtype=np.float64))
L2C=np.log10(df['L_2_Cubic'].to_numpy(dtype=np.float64))
LinfC=np.log10(df['L_inf_Cubic'].to_numpy(dtype=np.float64))
L2L=np.log10(df['L_2_Linear'].to_numpy(dtype=np.float64))
LinfL=np.log10(df['L_inf_Linear'].to_numpy(dtype=np.float64))


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return  b_1

L2C_Ord=estimate_coef(h,L2C)
LinfC_Ord=estimate_coef(h,LinfC)
L2L_Ord=estimate_coef(h,L2L)
LinfL_Ord=estimate_coef(h,LinfL)


print("L_2 Cubic Order",L2C_Ord)
print("L_inf Cubic Order",LinfC_Ord)
print("L_2 Linear Order",L2L_Ord)
print("L_inf Linear Order",LinfL_Ord)
