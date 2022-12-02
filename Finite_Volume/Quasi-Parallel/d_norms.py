from turtle import shape
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

def Norm_Time(x_vector,y_approx):
    int_L2=0
    int_L2_temp=0
    int_inf=0
    int_inf_temp=0
    int_inf_temp_2=0
    t=1 
    x_g=np.array([.77459667,0,-.77459667], dtype=np.float64)
    x_1=((x_vector[1:]-x_vector[:-1])/2)*x_g[0]+(((x_vector[1:]+x_vector[:-1])/2))
    x_2=((x_vector[1:]-x_vector[:-1])/2)*x_g[1]+(((x_vector[1:]+x_vector[:-1])/2))
    x_3=((x_vector[1:]-x_vector[:-1])/2)*x_g[2]+(((x_vector[1:]+x_vector[:-1])/2))
    x=np.zeros(shape=(3,len(x_1)),dtype=np.float64)
    x[0,:]=x_1
    x[1,:]=x_2
    x[2,:]=x_3
    w_g=np.array([5/9,8/9,5/9], dtype=np.float64)
    for i in range(len(x_vector)-1):
        interp_1=interp1d(x_vector[i:i+2].squeeze(),y_approx[i:i+2].squeeze(), kind='linear')
        for (j,w) in enumerate(w_g):
            true_int=True_Solution(x[j,i],t)
            approx=interp_1(x[j,i])
            int_L2_temp+=w*((true_int-approx)**2)
            int_inf_temp_2=abs(true_int-approx)
            if(int_inf_temp_2>int_inf_temp):
                int_inf_temp=int_inf_temp_2
        int_L2+=(int_L2_temp)*(x_vector[i+1]-x_vector[i])/2
        int_L2_temp=0
        if(int_inf_temp>int_inf):
            int_inf=int_inf_temp
        int_inf_temp=0
        int_inf_temp_2=0
        j=0
    int_L2=math.sqrt(int_L2)
    return int_L2,int_inf

def True_Solution(x,t):
    u_true= lambda ep,x,t: math.exp(-.01*t*math.cos(math.pi/10)*(ep**1.8))*math.cos(x*ep)
    int=(1/math.pi)*(quad(u_true,0,10**3,args=(x,t))[0]+quad(u_true,10**3,10**6,args=(x,t))[0]+quad(u_true,10**6,10**7,args=(x,t))[0]\
        +quad(u_true, 10**7,10**8, args=(x,t))[0]+quad(u_true,10**8,np.inf,args=(x,t))[0])
    return int

def Left_True_Solution(x_mesh):
    u_true=lambda x: x**2-x**3
    u=u_true(x_mesh)
    return u

def Norm_SS(x_vector,y_vector):
    int_inf=0
    int_inf_temp=0
    for i in range(len(x_vector)):
        int_inf_temp=abs(y_vector[i]-Left_True_Solution(x_vector[i]))
        if(int_inf_temp>int_inf):
            int_inf=int_inf_temp
    return int_inf