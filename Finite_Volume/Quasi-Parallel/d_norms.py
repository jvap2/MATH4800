from turtle import shape
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import quad, trapezoid
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

def True_Solution(x_sol):
    u_true= lambda ep,x: math.exp(-.01*math.cos(math.pi/10)*(ep**1.8))*math.cos(x*ep)
    int=np.empty(shape=(len(x_sol)))
    for (i,point) in enumerate(x_sol):
        int[i]=(1/math.pi)*(quad(u_true,0,10**4,args=(point))[0])+(1/math.pi)*(quad(u_true,10**4,10**8,args=(point))[0])+\
            (1/math.pi)*(quad(u_true,10**8,10**12,args=(point))[0])+(1/math.pi)*(quad(u_true,10**12,10**16,args=(point))[0])
    return int

def Left_True_Solution(x_mesh):
    u_true=lambda x: x**2-x**3
    u=u_true(x_mesh)
    return u

def Right_True_Solution(x_mesh):
    u_true=lambda x: x**3-x**2
    u=u_true(x_mesh)
    return u

def Left_Ex_1(x_mesh):
    u_true=lambda x: x**5-x**4
    return u_true(x_mesh)

def Left_Ex_2(x_mesh):
    u_true=lambda x: x**(4.12)-x
    return u_true(x_mesh)

def Left_Ex_3(x_mesh):
    u_true=lambda x: x**(3.63)-x**2
    return u_true(x_mesh)


def Norm_SS(x_vector,y_approx, type):
    int_L2=0
    int_L2_temp=0
    int_inf=0
    int_inf_temp=0
    int_inf_temp_2=0
    t=1 
    if type=='linear':
        order=1
    else:
        order=3
    # x_g=np.array([.77459667,0,-.77459667], dtype=np.float64)
    x_g=np.array([-.96028986,-.79666648,-.52553241,-.18343464,.18343464,.52553241,.79666648,.96028986])
    x_1=((x_vector[1:]-x_vector[:-1])/2)*x_g[0]+(((x_vector[1:]+x_vector[:-1])/2))
    x_2=((x_vector[1:]-x_vector[:-1])/2)*x_g[1]+(((x_vector[1:]+x_vector[:-1])/2))
    x_3=((x_vector[1:]-x_vector[:-1])/2)*x_g[2]+(((x_vector[1:]+x_vector[:-1])/2))
    x_4=((x_vector[1:]-x_vector[:-1])/2)*x_g[3]+(((x_vector[1:]+x_vector[:-1])/2))
    x_5=((x_vector[1:]-x_vector[:-1])/2)*x_g[4]+(((x_vector[1:]+x_vector[:-1])/2))
    x_6=((x_vector[1:]-x_vector[:-1])/2)*x_g[5]+(((x_vector[1:]+x_vector[:-1])/2))
    x_7=((x_vector[1:]-x_vector[:-1])/2)*x_g[6]+(((x_vector[1:]+x_vector[:-1])/2))
    x_8=((x_vector[1:]-x_vector[:-1])/2)*x_g[7]+(((x_vector[1:]+x_vector[:-1])/2))
    x=np.zeros(shape=(8,len(x_1)),dtype=np.float64)
    x[0,:]=x_1
    x[1,:]=x_2
    x[2,:]=x_3
    x[3,:]=x_4
    x[4,:]=x_5
    x[5,:]=x_6
    x[6,:]=x_5
    x[7,:]=x_6
    # w_g=np.array([5/9,8/9,5/9], dtype=np.float64)
    w_g=np.array([.110122854,.22238103,.31370665,.3626837833,.3626837833,.31370665,.22238103,.110122854], dtype=np.float64)
    for i in range(len(x_vector)-order):
        interp_1=interp1d(x_vector[i:i+(order+1)].squeeze(),y_approx[i:i+(order+1)].squeeze(), kind=type)
        for (j,w) in enumerate(w_g):
            true_int=Left_Ex_3(x[j,i])
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

def Norm_SS_Right(x_vector,y_vector):
    int_inf=0
    int_inf_temp=0
    for i in range(len(x_vector)):
        int_inf_temp=abs(y_vector[i]-Right_True_Solution(x_vector[i]))
        if(int_inf_temp>int_inf):
            int_inf=int_inf_temp
    return int_inf 


def CompTrad_Err_ThirdOrder(x_vector, y_approx):
    int_L_2=0
    int_inf_temp=0
    int_inf=0
    for i in range(1,len(x_vector)-2):
        if i==1:
            x_sample=np.linspace(x_vector[i], x_vector[i+1],20)
            y_inter_1=lambda x: (y_approx[i]/2)*(x-x_vector[i-1])*(x-x_vector[i+1])*(x-x_vector[i+2])
            y_inter_2=lambda x: (-y_approx[i+1]/2)*(x-x_vector[i-1])*(x-x_vector[i])*(x-x_vector[i+2])
            y_inter_3=lambda x: (y_approx[i+2]/6)*(x-x_vector[i])*(x-x_vector[i+1])*(x-x_vector[i-1])
            y_est=y_inter_1(x_sample)+y_inter_2(x_sample)+y_inter_3(x_sample)
            print(y_est)
            y_true=Left_Ex_3(x_sample)
            print(y_true)
            eval_L_2=(y_est-y_true)**2
            eval_L_inf=np.max(np.abs(y_true-y_est))
        elif i>=2 and i<len(x_vector)-2:
            x_sample=np.linspace(x_vector[i], x_vector[i+1],20)
            y_inter_0=lambda x: (-y_approx[i-2]/6)*(x-x_vector[i-1])*(x-x_vector[i+1])*(x-x_vector[i])
            y_inter_1=lambda x: (y_approx[i-1]/2)*(x-x_vector[i])*(x-x_vector[i+1])*(x-x_vector[i-2])
            y_inter_2=lambda x: (-y_approx[i]/2)*(x-x_vector[i-1])*(x-x_vector[i-2])*(x-x_vector[i-1])
            y_inter_3=lambda x: (y_approx[i+1]/6)*(x-x_vector[i])*(x-x_vector[i-1])*(x-x_vector[i-2])
            y_est=y_inter_0(x_sample)+y_inter_1(x_sample)+y_inter_2(x_sample)+y_inter_3(x_sample)
            y_true=Left_Ex_3(x_sample)
            eval_L_2=(y_est-y_true)**2
            eval_L_inf=np.max(np.abs(y_true-y_est))
        else:
            x_sample=np.linspace(x_vector[i], x_vector[i+1],20)
            y_inter_1=lambda x: (-y_approx[i-2]/6)*(x-x_vector[i-1])*(x-x_vector[i+1])*(x-x_vector[i])
            y_inter_2=lambda x: (y_approx[i-1]/2)*(x-x_vector[i])*(x-x_vector[i+1])*(x-x_vector[i-2])
            y_inter_3=lambda x: (-y_approx[i]/2)*(x-x_vector[i-1])*(x-x_vector[i-2])*(x-x_vector[i-1])
            y_est=y_inter_1(x_sample)+y_inter_2(x_sample)+y_inter_3(x_sample)
            y_true=Left_Ex_3(x_sample)
            eval_L_2=(y_est-y_true)**2
            eval_L_inf=np.max(np.abs(y_true-y_est))
        int_L_2+=trapezoid(eval_L_2,x_sample)
        int_inf_temp=eval_L_inf
        if int_inf_temp>int_inf:
            int_inf=int_inf_temp
    int_L_2=math.sqrt(int_L_2)
    return int_L_2, int_inf

        


