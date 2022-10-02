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
import scipy
from scipy.sparse.linalg import cg,gmres,cgs, minres
from scipy.linalg import norm
from numba import vectorize
from scipy.sparse import csc_matrix







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
        course=np.zeros((self.N,self.M))
        fine=np.zeros((self.N,self.M))
        course_temp=np.zeros((self.N,self.M))
        u=np.zeros((self.N,self.M+1))
        u_prev=np.zeros((self.N,self.M+1))
        u_temp=np.zeros((self.N))
        u_fine=np.zeros((self.N,self.M))
        k=0
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time_points()
        m=self.M
        fine_m=m-1
        N=self.N
        tol=9e-5
        d_t=self.mesh.delta_t()[0]/256
        b_temp=np.empty(shape=(N,1))
        B=self.stiff.B()
        M_1=self.mass.Construct_Prob_1_Init()
        M_t=self.mass.Construct()
        M_t_inv=np.linalg.inv(M_t)
        F=self.force.Construct()
        Mat_1=np.matmul(M_t_inv,d_t*B)+np.identity(N)
        Mat_2=np.matmul(M_t_inv,d_t*F)
        d_M_1=cuda.to_device(Mat_1)
        d_M_2=cuda.to_device(Mat_2)
        while error>tol:
            u[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                if i==1:
                    temp=np.matmul((M_1+(1-self.theta)*self.mesh.delta_t()[j]*B),u[:,j])+self.mesh.delta_t()[j]*F
                    b_temp=np.reshape(temp,newshape=(N))
                else:
                    temp=np.matmul((M_t+(1-self.theta)*self.mesh.delta_t()[j]*B),u[:,j])+self.mesh.delta_t()[j]*F
                    b_temp=np.reshape(temp,newshape=(N))
                A=(M_t-(self.theta)*self.mesh.delta_t()[j]*B)
                x,exit_code=cgs(A,b_temp)
                if exit_code!=0:
                    print("Failed Convergence")
                    break
                else:
                    course[:,j]=x                
                u[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
            course_temp=course.copy()
            u_fine[:,0]=np.linalg.solve(M_1-(.5)*d_t*B,np.matmul(M_1+(.5)*d_t*B,u_temp)+d_t*F)
            u_temp=u_fine[:,0]
            for i in range(255):
                u_fine[:,0]=np.matmul(Mat_1,u_temp)+Mat_2
                u_temp=u_fine[:,0]
            threads_per_block=64
            blocks_per_grid=(fine_m+threads_per_block-1)//threads_per_block
            d_u=cuda.to_device(np.ascontiguousarray(u[:,1:]))
            d_u_temp=cuda.to_device(u_fine)
            P1_fine[blocks_per_grid,threads_per_block](d_M_1,d_M_2,d_u,d_u_temp,N,fine_m)
            fine=d_u_temp.copy_to_host()
            error=norm(u_prev-u, ord=np.inf)
            u_prev=u.copy()
            k=k+1
            fine[:,0]=u_fine[:,0]
            print(k)
            print(error)
        return u
    def Parareal(self):
        m=self.M
        N=self.N
        course=np.zeros((self.N,self.M))
        fine=np.zeros((self.N,self.M))
        course_temp=np.zeros((self.N,self.M))
        u=np.zeros((self.N,self.M+1))
        u_temp=np.zeros((self.N,self.M+1))
        u_fine=np.zeros((self.N,self.M+1))
        k=0
        error=1
        u_0=self.u_zero(self.mesh.mesh_points()[1:N+1])
        t=self.mesh.time_points()
        tol=1e-6
        b_temp=np.empty(shape=(N,1))
        A=np.empty(shape=(N,N))
        B=lambda t: self.stiff.B_P_2(t)
        fine_size=8*m+1
        d_t=(t[1]-t[0])/8
        M_t=self.mass.Construct()
        F=self.force.Construct()
        M_t_inv=np.linalg.inv(M_t)
        M_1=np.empty((fine_size,N,N))
        B_t=np.empty((fine_size,N,N))
        for i in range(fine_size):
            B_t[i,:,:]=B(i*d_t)
            M_1[i,:,:]=np.matmul(M_t_inv,d_t*B_t[i,:,:])+np.identity(N)
        M_2=d_t*F
        d_M_1=cuda.to_device(M_1)
        d_M_2=cuda.to_device(M_2)
        while error>tol and k<20:
            u[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                b_temp=np.matmul((M_t+(1-self.theta)*self.mesh.delta_t()[j]*B(t[j])),u[:,j])+self.mesh.delta_t()[j]*F
                A=(M_t-(self.theta)*self.mesh.delta_t()[j]*B(t[i]))
                x,exit_code=cgs(A=A,b=b_temp, x0=u[:,j])
                if exit_code!=0:
                    print("Failed Convergence")
                    course[:,j]=np.random.rand(N)
                else:
                    course[:,j]=x
                u[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
                u_fine[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
            course_temp=course
            threads_per_block=m
            blocks_per_grid=(m+threads_per_block-1)//threads_per_block
            fine=np.zeros(shape=(self.N,self.M))
            d_u=cuda.to_device(u_fine)
            d_u_temp=cuda.to_device(fine)
            Fine[blocks_per_grid,threads_per_block](d_M_1,d_M_2,d_u,d_u_temp,N,m)
            cuda.synchronize()
            d_u_temp.copy_to_host(fine)
            error=norm(u_temp-u)
            u_temp=u
            k=k+1
            print("k:\n",k)
            print("Error:\n",error)
        return u
            



@cuda.jit("void(float64[:,:],float64[:],float64[:,:],float64[:,:],float64,float64)")
def P1_fine(M_1,M_2,u,u_temp,Nsize,Msize):
    idx_x=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    fSum=0
    if idx_x<Msize:
        for i in range(256):
            for j in range(Nsize):
                for k in range(Nsize):
                    fSum+=M_1[j,k]*u[k,idx_x+1]
                u_temp[j,idx_x+1]=fSum
                fSum=0
                u_temp[j,idx_x+1]+=M_2[j]
                cuda.syncthreads()
            for m in range(Nsize):
                u[m,idx_x]=u_temp[m,idx_x+1]

@cuda.jit("void(float64[:,:,:],float64[:],float64[:,:],float64[:,:],int32,int32)")
def Fine(M_1,M_2,u,u_temp,Nsize,Msize):
    idx_x=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    mat_ind=idx_x*8
    print(idx_x)
    if idx_x<Msize:
        print(idx_x)
        for i in range(8):
            for j in range(Nsize):
                for k in range(Nsize):
                    u_temp[j,idx_x]+=M_1[mat_ind+i,j,k]*u[k,idx_x]
                    cuda.syncthreads()
                u_temp[j,idx_x]+=M_2[j]
            cuda.syncthreads()
            for m in range(Nsize):
                u[m,idx_x]=u_temp[m,idx_x]
            cuda.syncthreads()




            







