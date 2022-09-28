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
        next_fine=np.zeros((self.N,self.M-1))
        fine_u=np.empty((self.N,self.M-1))
        course_temp=np.zeros((self.N,self.M))
        u=np.zeros((self.N,self.M+1))
        u_temp=np.zeros((self.N,self.M+1))
        k=0
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time_points()
        m=self.M
        N=self.N
        tol=9e-5
        A=np.zeros(shape=(N,N))
        A_temp=np.empty(shape=(N,N))
        A_inv=np.zeros(shape=(m,N,N))
        b=np.zeros(shape=(N,m))
        b_temp=np.empty(shape=(N,1))
        B=self.stiff.B()
        M_1=self.mass.Construct_Prob_1_Init()
        M_t=self.mass.Construct()
        M_t_inv=np.linalg.inv(M_t)
        F=self.force.Construct()
        while error>tol and k<30:
            u[:,0]=u_0
            fine[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                if i==1:
                    temp=np.matmul((M_1+(1-self.theta)*self.mesh.delta_t()[j]*B),u[:,j])+self.mesh.delta_t()[j]*F
                    b_temp=np.reshape(temp,newshape=(N))
                else:
                    temp=np.matmul((M_t+(1-self.theta)*self.mesh.delta_t()[j]*B),u[:,j])+self.mesh.delta_t()[j]*F
                    b_temp=np.reshape(temp,newshape=(N))
                A=(M_t-(self.theta)*self.mesh.delta_t()[j]*B)
                x,exit_code=cgs(A=A,b=b_temp, x0=u[:,j])
                if exit_code!=0:
                    print("Failed Convergence")
                    break
                else:
                    course[:,j]=np.reshape(x,newshape=(N))
                u[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
            course_temp[:,:]=course[:,:]
            fine_u=np.ascontiguousarray(u[:,1:])
            d_M=cuda.to_device(M_t)
            d_M_inv=cuda.to_device(M_t_inv)
            d_B=cuda.to_device(B)
            d_f=cuda.to_device(F)
            d_fine=cuda.to_device(fine_u)
            d_fine_next=cuda.to_device(next_fine)
            d_t=cuda.to_device(t)
            threads_per_block=(32,32,32)
            blocks_per_grid=((N+threads_per_block[0]-1)//threads_per_block[0],(N+threads_per_block[1]-1)//threads_per_block[1],(m+threads_per_block[2]-1)//threads_per_block[2])
            fine_P1[blocks_per_grid,threads_per_block](d_M,d_M_inv,d_B,d_f,d_t,d_fine,d_fine_next,N,(m-1))
            d_fine_next.copy_to_host(fine)
            u_temp[:,:]=u[:,:]
            k=k+1
            print(k)
            print(error)
        return u
    def Parareal(self):
        course=np.zeros((self.N,self.M))
        fine=np.zeros((self.N,self.M))
        course_temp=np.zeros((self.N,self.M))
        u=np.zeros((self.N,self.M+1))
        u_temp=np.zeros((self.N,self.M+1))
        k=0
        error=1
        x=self.mesh.mesh_points()[1:-1]
        u_0=self.u_zero(x)
        t=self.mesh.time_points()
        m=self.M
        N=self.N
        tol=5e-1
        b=np.zeros(shape=(N,m))
        b_temp=np.empty(shape=(N,1))
        A=np.zeros(shape=(N,N))
        A_temp=np.empty(shape=(N,N))
        A_inv=np.zeros(shape=(m,N,N))
        B=lambda t: self.stiff.B_P_2(t)
        M_t=self.mass.Construct()
        F=self.force.Construct()
        while error>tol and k<40:
            u[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                temp=np.matmul((M_t+(1-self.theta)*self.mesh.delta_t()[j]*B(t[j])),u[:,j])+self.mesh.delta_t()[j]*F
                b_temp=np.reshape(temp,newshape=(N))
                b[:,j]=b_temp
                A=(M_t-(self.theta)*self.mesh.delta_t()[j]*B(t[i]))
                A_inv[j,:,:]=np.linalg.inv(A)
                x,exit_code=cgs(A=A,b=b_temp, x0=u[:,j])
                if exit_code!=0:
                    print("Failed Convergence")
                    break
                else:
                    course[:,j]=np.reshape(x,newshape=(N))
                u[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
            course_temp[:,:]=course[:,:]
            u_temp[:,:]=u[:,:]
            k=k+1
            print(k)
            print(error)
        return u
            


@cuda.jit("void(float64[:,:],float64[:,:],float64[:,:],float64[:],float64[:],float64[:,:],float64[:,:], float64,float64)")
def fine_P1(M,M_inv,B,f,t,u,u_prev, Nsize,Msize):
    idx_x=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    idx_y=cuda.threadIdx.y+(cuda.blockDim.y*cuda.blockIdx.y)
    idx_z=cuda.threadIdx.z+(cuda.blockDim.z*cuda.blockIdx.z)
    fSum=0
    if idx_z<Msize:
        t_0,t_1=t[idx_z+1],t[idx_z+2]
        delta_t=(t_1-t_0)/5
    if idx_x<Nsize and idx_y<Nsize:
        u_prev[idx_x,idx_z]=u[idx_x,idx_z]
        for i in range(5):
            for j in range(Nsize):
                fSum+=M_inv[idx_x,j]*((M[j,idx_y]+(t_0*delta_t*i)*B[j,idx_y]*u_prev[j,idx_z])+(t_0*delta_t*i)*f[j])
        u[idx_x,idx_z]=fSum







