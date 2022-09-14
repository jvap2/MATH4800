from ast import arg
from re import M
from d_mass_matrix import MassMatrix
from d_force_matrix import Force_Matrix
from d_stiff_matrix import StiffMatrix
from d_mesh import Mesh
import cupyx.scipy
from cupyx.scipy.sparse import csc_matrix, linalg
import cupy as cp
from scipy.integrate import quad
import numpy as np
import math
import time




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
        return cp.exp(-(x-1)**2/(2*.08**2))
    def u_zero_1(self):
        u=cp.zeros(self.N)
        u[self.N//2]=1
        return u
    def sol_1(self,x,t):
        u_true= lambda ep,x,t: math.exp(-.01*t*math.cos(math.pi/10)*(ep**1.8))*math.cos(x*ep)
        int=(1/math.pi)*(quad(u_true,0,10**3,args=(x,t))[0]+quad(u_true,10**3,10**6,args=(x,t))[0]+quad(u_true,10**6,10**7,args=(x,t))[0]\
            +quad(u_true, 10**7,10**8, args=(x,t))[0]+quad(u_true,10**8,np.inf,args=(x,t))[0])
        return int
    def CGS(self):
        u_0=self.u_zero_1()
        u=cp.zeros((self.N,self.M+1))
        u[:,0]=u_0
        for (i,t) in enumerate(self.mesh.time()[1:]):
            if i==0:
                b=cp.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            else:
                b=cp.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            A=csc_matrix(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B(t))
            x,exit_code=linalg.cgs(A,b)
            if exit_code !=0:
                print("Failed convergence")
                break
            else:
                u[:,i+1]=x
            if(i%10==0):
                print("Iteration:",i)
        return cp.asnumpy(u)
    def MatInv(self):
        u_0=self.u_zero_1()
        u=cp.zeros((self.N,self.M+1))
        u[:,0]=u_0
        for (i,t) in enumerate(self.mesh.time()[1:]):
            if i==0:
                b=cp.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            else:
                b=cp.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            A=(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B(t))
            u[:,i+1]=cp.matmul(cp.linalg.inv(A),b)
            if(i%10==0):
                print("Iteration:",i)
        return cp.asnumpy(u)
    def True_Sol(self):
        u_true_final=cp.zeros((self.N+2))
        for (i,x) in enumerate(self.mesh.mesh_points()):
            u_true_final[i]=self.sol_1(x,t=1)
        return cp.asnumpy(u_true_final)


