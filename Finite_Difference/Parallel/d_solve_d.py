import d_mesh_d,d_nsprmat_d,d_sink_d
import cupy as cp
import cupyx
import cupyx.scipy
from cupyx.scipy.sparse import csc_matrix, linalg
import cupy as cp
from scipy.integrate import quad
import numpy as np
import math


class FD_Solve():
    def __init__(self,a,b,N,t_0,t_m,M,alpha,gamma):
        self.mesh=d_mesh_d.Mesh(a,b,N,t_0,t_m,M)
        self.sink=d_sink_d.Sink_Matrix(a,b,N,t_0,t_m,M)
        self.B=d_nsprmat_d.B_mat(a,b,N,t_0,t_m,M,alpha,gamma)
        self.N=N
        self.M=M
    def u_init(self):
        u_0=cp.zeros(self.N+1)
        u_0[(self.N)//2]=1
        return u_0
    def dirac_B_mat(self):
        B=cp.zeros(shape=(self.N+1,self.N+1))
        B[self.N/2,self.N/2]=1
        return B
    def sol_1(self,x,t):
        u_true= lambda ep,x,t: math.exp(-.01*t*math.cos(math.pi/10)*(ep**1.8))*math.cos(x*ep)
        int=(1/math.pi)*(quad(u_true,0,10**3,args=(x,t))[0]+quad(u_true,10**3,10**6,args=(x,t))[0]+quad(u_true,10**6,10**7,args=(x,t))[0]\
            +quad(u_true, 10**7,10**8, args=(x,t))[0]+quad(u_true,10**8,np.inf,args=(x,t))[0])
        return int
    def sol(self):
        u=cp.zeros((self.N+1,self.M+1))
        u[:,0]=self.u_init()
        for (i,t) in enumerate(self.mesh.time()[0:self.M]):
            if i==0:
                u[:,i+1]=cp.matmul(self.dirac_B_mat(),u[:,0])+self.mesh.delta_t()*self.sink.Construct()
            else:
                u[:,i+1]=cp.matmul(self.B.Construct(),u[:,i])+self.mesh.delta_t()*self.sink.Construct()
        return cp.asnumpy(u)
    def true_sol(self):
        u_true_final=cp.zeros((self.N+1))
        u_true_mid=cp.zeros((self.N+1))
        for (i,x) in enumerate(self.mesh.mesh_points()):
            u_true_final[i]=self.sol_1(x,t=1)
            u_true_mid[i]=self.sol_1(x,t=.5)
        return cp.asnumpy(u_true_final),cp.asnumpy(u_true_mid)