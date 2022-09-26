from ast import arg
from concurrent.futures import thread
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
        self.force=Force_Matrix(N)
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
        fine=np.zeros((self.N,self.M+1))
        u=np.zeros((self.N,self.M+1))
        u_temp=np.zeros((self.N,self.M+1))
        k=0
        tol=5e-5
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time_points()
        x=self.mesh.mesh_points()
        m=self.M
        N=self.N
        b=np.empty(shape=(N,1,m+1))
        A=np.empty(shape=(N,N,m+1))
        A_inv=np.empty(shape=(N,N,m+1))
        while error>tol:
            u[:,0]=u_0
            fine[:,0]=u_0
            for i in range(1,m):
                if i==1:
                    temp=np.matmul((self.mass.Construct_Prob_1_Init()+(1-self.theta)*self.mesh.delta_t()[i-1]*self.stiff.B(self.mesh.time_points()[i-1])),u[:,i-1])+self.mesh.delta_t()[i-1]*self.force.Construct()
                    b[:,:,i]=np.reshape(temp,newshape=(N,1))
                else:
                    temp=np.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()[i-1]*self.stiff.B(self.mesh.time_points()[i-1])),u[:,i-1])+self.mesh.delta_t()[i-1]*self.force.Construct()
                    b[:,:,i]=np.reshape(temp,newshape=(N,1))
                A[:,:,i]=(self.mass.Construct()-(self.theta)*self.mesh.delta_t()[i-1]*self.stiff.B(t[i]))
                A_inv[:,:,i]=np.linalg.inv(A[:,:,i])
                course[:,i]=np.reshape(np.matmul(A_inv[:,:,i],b[:,:,i]),newshape=(N))
                u[:,i]=fine[:,i]+course[:,i]-course[:,i-1]
            d_A=cuda.to_device(A_inv)
            d_b=cuda.to_device(b)
            d_f=cuda.to_device(fine)
            threads_per_block=(16,16)
            blocks_per_grid=((m+threads_per_block[0])//threads_per_block[0],(N+threads_per_block[1])//threads_per_block[1])
            print(np.shape(A_inv))
            print(np.shape(b))
            calculate_fine[blocks_per_grid,threads_per_block](d_A,d_b,d_f,N,M+1)
            d_f.copy_to_host(fine)
            d_b.copy_to_host(b)
            d_A.copy_to_host(A_inv)
            error=np.max(np.linalg.norm(u-u_temp))
            u_temp=u
        return u

            


@cuda.jit("void(float64[:,:,:],float64[:,:,:],float64[:,:],float64,float64)")
def calculate_fine(A,b,fine,Nsize,Msize):
    idx_x=cuda.threadIdx.x+(cuda.blockIdx.x*cuda.blockDim.x)
    idx_y=cuda.threadIdx.y+(cuda.blockIdx.y*cuda.blockDim.y)
    fSum=0
    if idx_x<Msize and idx_y<Nsize:
        for i in range(Nsize):
            fSum+=A[idx_y,i,idx_x]*b[i,0,idx_x]
    fine[idx_y+1,idx_x]=fSum
            
f=Final_Solution(-4,4,127,0,1,128,.5,.2,.5)
f.Parareal_1()



