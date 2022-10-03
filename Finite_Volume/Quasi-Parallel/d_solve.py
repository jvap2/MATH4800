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
import numba
from numba import cuda
from joblib import Parallel,delayed
from scipy.linalg import norm
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
        return cp.exp(-(x-1)**2/(2*.08**2))
    def u_zero_1(self):
        u=cp.zeros(self.N, dtype=cp.float16)
        u[self.N//2]=1.0
        return u
    def sol_1(self,x,t):
        u_true= lambda ep,x,t: math.exp(-.01*t*math.cos(math.pi/10)*(ep**1.8))*math.cos(x*ep)
        int=(1/math.pi)*(quad(u_true,0,10**3,args=(x,t))[0]+quad(u_true,10**3,10**6,args=(x,t))[0]+quad(u_true,10**6,10**7,args=(x,t))[0]\
            +quad(u_true, 10**7,10**8, args=(x,t))[0]+quad(u_true,10**8,np.inf,args=(x,t))[0])
        return int
    def CGS(self):
        mempool = cp.get_default_memory_pool()
        u_0=self.u_zero_1()
        u=np.zeros((self.N,self.M+1),dtype=np.float32)
        u_temp=u_0
        u[:,0]=cp.asnumpy(u_0)
        for (i,t) in enumerate(self.mesh.time()[1:]):
            if i==0:
                b=cp.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u_temp,dtype=cp.float16)+self.mesh.delta_t()*self.force.Construct()
            else:
                b=cp.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u_temp,dtype=cp.float16)+self.mesh.delta_t()*self.force.Construct()
            A=csc_matrix(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B(t), dtype=cp.float32)
            x,exit_code=linalg.cgs(A=A,b=b, x0=u_temp)
            if exit_code !=0:
                print("Failed convergence")
                break
            else:
                u_temp=x
                u[:,i+1]=cp.asnumpy(u_temp)
            if(i%10==0):
                print("Iteration:",i)
        print(f"Used bytes before: {mempool.used_bytes()}")
        print(f"Total_bytes before: {mempool.total_bytes()}")
        mempool.free_all_blocks()
        print(f"Used bytes after: {mempool.used_bytes()}")
        print(f"Total_bytes after: {mempool.total_bytes()}")
        return u
    def MatInv(self):
        mempool = cp.get_default_memory_pool()
        u_0=self.u_zero_1()
        u=cp.zeros((self.N,self.M+1),dtype=cp.float16)
        u[:,0]=u_0
        for (i,t) in enumerate(self.mesh.time()[1:]):
            if i==0:
                b=cp.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i], dtype=cp.float16)+self.mesh.delta_t()*self.force.Construct()
            else:
                b=cp.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i], dtype=cp.float16)+self.mesh.delta_t()*self.force.Construct()
            A=(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B(t))
            u[:,i+1]=cp.matmul(cp.linalg.inv(A),b)
            if(i%10==0):
                print("Iteration:",i)
        u_sol=cp.asnumpy(u)
        print(f"Used bytes before: {mempool.used_bytes()}")
        print(f"Total_bytes before: {mempool.total_bytes()}")
        mempool.free_all_blocks()
        print(f"Used bytes after: {mempool.used_bytes()}")
        print(f"Total_bytes after: {mempool.total_bytes()}")
        return u_sol
    def True_Sol(self):
        mempool = cp.get_default_memory_pool()
        u_true_final=cp.zeros((self.N+2))
        for (i,x) in enumerate(self.mesh.mesh_points()):
            u_true_final[i]=self.sol_1(x,t=1)
        u=cp.asnumpy(u_true_final)
        print(f"Used bytes before: {mempool.used_bytes()}")
        print(f"Total_bytes before: {mempool.total_bytes()}")
        mempool.free_all_blocks()
        print(f"Used bytes after: {mempool.used_bytes()}")
        print(f"Total_bytes after: {mempool.total_bytes()}")
        return u
    def Parareal(self):
        m=self.M
        N=self.N
        course=cp.zeros((self.N,self.M), dtype=cp.float32)
        fine=cp.zeros((self.N,self.M), dtype=cp.float32)
        course_temp=cp.zeros((self.N,self.M), dtype=cp.float32)
        u=cp.zeros((self.N,self.M+1), dtype=cp.float32)
        u_temp=cp.zeros((self.N,self.M+1), dtype=cp.float32)
        u_fine=cp.zeros((self.N,self.M+1), dtype=cp.float32)
        k=0
        error=1
        u_0=self.u_zero(self.mesh.mesh_points()[1:N+1])
        t=self.mesh.time()
        tol=1e-6
        b_temp=cp.empty(shape=(N,1), dtype=cp.float32)
        A=cp.empty(shape=(N,N), dtype=cp.float32)
        B=lambda t: self.stiff.B(t)
        fine_m=4096//m
        d_t=(t[1]-t[0])/fine_m
        M_t=self.mass.Construct()
        F=self.force.Construct()
        M_t_inv=cp.linalg.inv(M_t)
        M_1=cp.empty((m,fine_m,N,N), dtype=cp.float32)
        B_t=cp.empty((m,fine_m,N,N), dtype=cp.float32)
        for i in range(m):
            for j in range(fine_m):
                B_t[i,j,:,:]=B((i*fine_m+j)*d_t)
                M_1[i,j,:,:]=cp.matmul(M_t_inv,d_t*B_t[i,j,:,:])+cp.identity(N)
        print(cp.shape(M_1))
        M_2=d_t*F
        while error>tol or k<20:
            u[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                b_temp=cp.matmul((M_t+(1-self.theta)*self.mesh.delta_t()*B(t[j])),u[:,j])+self.mesh.delta_t()*F
                A=csc_matrix(M_t-(self.theta)*self.mesh.delta_t()*B(t[i]), dtype=cp.float32)
                x,exit_code=linalg.cgs(A=A,b=b_temp, x0=u[:,j])
                if exit_code!=0:
                    print("Failed Convergence")
                    course[:,j]=cp.random.rand(N)
                else:
                    course[:,j]=x
                u[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
            course_temp=course
            u_fine=u
            fine=cp.zeros(shape=(self.N,self.M))
            fine=cp.transpose(Parallel(n_jobs=-1,verbose=1)\
                (delayed(Fine_Propogator)(M_1[i],M_2,u_fine[:,i],fine[:,i],fine_m) for i in range(m)))
            print(cp.shape(fine))
            error=norm(u-u_temp)
            u_temp=u
            k+=1
            print(k)
            print(error)
        return u
    def Parareal_1(self):
        m=self.M
        N=self.N
        course=cp.zeros((self.N,self.M), dtype=cp.float32)
        fine=cp.zeros((self.N,self.M), dtype=cp.float32)
        course_temp=cp.zeros((self.N,self.M), dtype=cp.float32)
        u=cp.zeros((self.N,self.M+1), dtype=cp.float32)
        u_temp=cp.zeros((self.N,self.M+1), dtype=cp.float32)
        u_fine=cp.zeros((self.N,self.M+1), dtype=cp.float32)
        dif=cp.zeros((self.N,self.M+1), dtype=cp.float32)
        k=0
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time()
        tol=1e-8
        b_temp=cp.empty(shape=(N,1), dtype=cp.float32)
        A=cp.empty(shape=(N,N), dtype=cp.float32)
        B=self.stiff.B_1()
        fine_m=8192//m
        d_t=(t[1]-t[0])/fine_m
        M_t=self.mass.Construct()
        F=self.force.Construct()
        M_1_construct=self.mass.Construct_Prob_1()
        M_1_inv=cp.linalg.inv(M_1_construct)
        M_t_inv=cp.linalg.inv(M_t)
        M_1=cp.empty((m,fine_m,N,N), dtype=cp.float32)
        M_1_for_device=cp.empty(shape=(m*fine_m*N*N), dtype=cp.float32)
        for i in range(m):
            for j in range(fine_m):
                if i==0:
                    M_1[i,j,:,:]=cp.matmul(M_1_inv,d_t*B)+cp.identity(N)
                else:
                    M_1[i,j,:,:]=cp.matmul(M_t_inv,d_t*B)+cp.identity(N)
                    # for k in range(N):
                    #     for l in range(N):
                    #         M_1_for_device[i*fine_m*N*N+j*N*N+k*N+l]=M_1[i,j,k,l]
        M_2=d_t*F
        M_1_for_device=M_1.flatten()
        M_2_for_device=M_2.flatten()
        while error>tol or k<30:
            u[:,0]=u_0
            b_temp=cp.matmul((M_1_construct+(1-self.theta)*self.mesh.delta_t()*B),u[:,0])+self.mesh.delta_t()*F
            A=csc_matrix(M_t-(self.theta)*self.mesh.delta_t()*B, dtype=cp.float32)
            x,exit_code=linalg.cgs(A=A,b=b_temp, x0=u[:,0],tol=5e-4)
            if exit_code!=0:
                print("Failed Convergence")
            else:
                course[:,0]=x
            u[:,1]=fine[:,0]+course[:,0]-course_temp[:,0]
            course_temp[:,0]=course[:,0]
            for i in range(2,m+1):
                b_temp=cp.matmul((M_t+(1-self.theta)*self.mesh.delta_t()*B),u[:,i-1])+self.mesh.delta_t()*F
                A=csc_matrix(M_t-(self.theta)*self.mesh.delta_t()*B)
                x,exit_code=linalg.cgs(A=A,b=b_temp, x0=u[:,i-1],tol=5e-4)
                if exit_code!=0:
                    print("Failed Convergence")
                else:
                    course[:,i-1]=x
                u[:,i]=fine[:,i-1]+course[:,i-1]-course_temp[:,i-1]
                course_temp[:,i-1]=course[:,i-1]
            u_fine=u
            fine_for_device=cp.zeros(shape=(1,self.N*self.M))
            u_fine_for_device=u_fine.flatten()
            threads_per_block=(128,)
            blocks_per_grid=((m+128-1)//128,)
            Fine_Propogator(blocks_per_grid,threads_per_block,(M_1_for_device,M_2_for_device,u_fine_for_device,fine_for_device,fine_m,N,M))
            fine=cp.reshape(u_fine_for_device,newshape=(N,m))
            dif=u-u_temp
            error=cp.max(dif)
            u_temp=u
            k+=1
            print(k)
            print(error)
        return cp.asnumpy(u)
    

Fine_Propogator=cp.RawKernel(r'''
extern "C" __global___
void fine_propogator(float* M_1, float* M_2, float* u, float* u_fine, int iter,int N, int M){
    idx_x=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx_x<M){
        for(int a{};a<iter;a++){
            for(int i{}; i<N;i++){
                for (int k{}; k<N;k++){
                        u_temp[i*N+idx]+=M_1[idx_x*fine_m*N*N+iter*N*N+i*N+k]*u[k*N+idx];
                }
                u_temp[i*N+idx]+=M_2[i*N+idx];
            }
            for(j{}; j<N;j++){
                u[j*N+idx]=u_temp[j*N+idx];
                u_temp[j*N+idx]=0;
            }
        }
    }
}
''','fine_propogator')