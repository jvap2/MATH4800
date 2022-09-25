from ast import arg
from re import M
from c_mass_matrix import MassMatrix
from c_force import Force_Matrix
from c_stiff_matrix import StiffMatrix
from c_mesh import Mesh
import cupy as cp
from scipy.integrate import quad
import numpy as np
import math
import time
import numba
from numba import cuda




class Final_Solution():
    def __init__(self,a,b,N,t_0,t_m,M,gamma,beta,theta):
        self.mass=MassMatrix(a,b,N,t_0, t_m,M)
        self.force=Force_Matrix(a,b,N,t_0,t_m,M)
        self.stiff=StiffMatrix(a,b,N,t_0,t_m,M,gamma,beta)
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.theta=theta
        self.N=N
        self.M=M
    def u_zero(self,x,t=0):
        return np.exp(-(x-1)**2/(2*.08**2))
    def u_zero_1(self):
        u=np.zeros(self.N, dtype=cp.float16)
        u[self.N//2]=1.0
        return u
    def sol_1(self,x,t):
        u_true= lambda ep,x,t: math.exp(-.01*t*math.cos(math.pi/10)*(ep**1.8))*math.cos(x*ep)
        int=(1/math.pi)*(quad(u_true,0,10**3,args=(x,t))[0]+quad(u_true,10**3,10**6,args=(x,t))[0]+quad(u_true,10**6,10**7,args=(x,t))[0]\
            +quad(u_true, 10**7,10**8, args=(x,t))[0]+quad(u_true,10**8,np.inf,args=(x,t))[0])
        return int
    def Parareal_1(self):
        course=np.zeros((self.N,self.M+1))
        course_temp=np.zeros((self.N,self.M+1))
        fine=np.zeros(self.N)
        fine_temp=np.zeros(self.N)
        u=np.empty((self.N,self.M+1))
        k=0
        tol=5e-5
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time_points()
        m=self.M
        while error>tol:
            u[:,0]=u_0
            for i in range(1,m):
                if i==0:
                    b=cp.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i-1])),u[:,i-1], dtype=cp.float16)+self.mesh.delta_t()*self.force.Construct()
                else:
                    b=cp.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i-1])),u[:,i-1], dtype=cp.float16)+self.mesh.delta_t()*self.force.Construct()
                A=(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B(t[i]))
                course[:,i]=cp.matmul(cp.linalg.inv(A),b)
                u[:,i]=fine[:,i]+course[:,i]-course[:,i-1]

