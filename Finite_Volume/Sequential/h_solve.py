from re import M
from h_mass_matrix import MassMatrix
from h_force_matrix import Force_Matrix
from h_stiff_matrix import StiffMatrix
from h_mesh import Mesh
import scipy
from scipy.sparse import csc_matrix
import numpy as np
from scipy.integrate import quad
import numpy as np
import math
import time
from scipy.linalg import norm
import numba
from numba import cuda
from scipy.sparse.linalg import cg,gmres,cgs, minres
from joblib import Parallel,delayed



class Final_Solution():
    def __init__(self,a,b,N,t_0,t_m,M,gamma,beta,theta):
        self.mass=MassMatrix(a,b,N,t_0, t_m,M)
        self.force=Force_Matrix(a,b,N,t_0,t_m,M)
        self.stiff=StiffMatrix(a,b,N,t_0,t_m,M,gamma,beta)
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.theta=theta
        self.N=N
        self.M=M
    def u_zero_1(self):
        u=np.zeros(self.N)
        u[self.N//2]=1
        return u
    def sol_1(self,x,t):
        u_true= lambda ep,x,t: math.exp(-.01*t*math.cos(math.pi/10)*(ep**1.8))*math.cos(x*ep)
        int=(1/math.pi)*(quad(u_true,0,10**3,args=(x,t))[0]+quad(u_true,10**3,10**6,args=(x,t))[0]+quad(u_true,10**6,10**7,args=(x,t))[0]\
            +quad(u_true, 10**7,10**8, args=(x,t))[0]+quad(u_true,10**8,np.inf,args=(x,t))[0])
        return int
    def u_zero(self,x):
        return np.exp(-(x-1)**2/(2*.08**2))
    def CGS(self):
        u_0=self.u_zero_1()
        u=np.zeros((self.N,self.M+1))
        u[:,0]=u_0
        for (i,t) in enumerate(self.mesh.time()[1:]):
            if i==0:
                b=np.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            else:
                b=np.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            A=csc_matrix(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B(t))
            x,exit_code=scipy.sparse.linalg.cgs(A,b)
            if exit_code !=0:
                print("Failed convergence")
                break
            else:
                u[:,i+1]=x
        return u
    def True_Sol(self):
        u_true_final=np.zeros((self.N+2))
        for (i,x) in enumerate(self.mesh.mesh_points()):
            u_true_final[i]=self.sol_1(x,t=1)
        return u_true_final
    def Parareal(self):
        m=self.M
        N=self.N
        course=np.zeros((self.N,self.M))
        fine=np.zeros((self.N,self.M))
        course_temp=np.zeros((self.N,self.M))
        u=np.zeros((self.N,self.M+1))
        u_temp=np.zeros((self.N,self.M+1))
        k=0
        error=1
        u_0=self.u_zero(self.mesh.mesh_points()[1:N+1])
        t=self.mesh.time()
        tol=1e-6
        b_temp=np.empty(shape=(N,1))
        A=np.empty(shape=(N,N))
        B=lambda t: self.stiff.B(t)
        fine_m=8
        d_t=(t[1]-t[0])/8
        M_t=self.mass.Construct()
        F=self.force.Construct()
        M_t_inv=np.linalg.inv(M_t)
        M_1=np.empty((m,fine_m,N,N))
        B_t=np.empty((m,fine_m,N,N))
        for i in range(m):
            for j in range(fine_m):
                B_t[i,j,:,:]=B((i*fine_m+j)*d_t)
                M_1[i,j,:,:]=np.matmul(M_t_inv,d_t*B_t[i,j,:,:])+np.identity(N)
        print(np.shape(M_1))
        M_2=d_t*F
        while error>tol and k<20:
            u[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                b_temp=np.matmul((M_t+(1-self.theta)*self.mesh.delta_t()*B(t[j])),u[:,j])+self.mesh.delta_t()*F
                A=(M_t-(self.theta)*self.mesh.delta_t()*B(t[i]))
                x,exit_code=cgs(A=A,b=b_temp, x0=u[:,j])
                if exit_code!=0:
                    print("Failed Convergence")
                    course[:,j]=np.random.rand(N)
                else:
                    course[:,j]=x
                u[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
            course_temp=course
            fine=np.zeros(shape=(self.N,self.M))
            print(np.shape(fine))
            fine=np.transpose(Parallel(n_jobs=-1,verbose=1)\
                (delayed(Fine_Propogator)(M_1[i],M_2,u[:,i],fine[:,i]) for i in range(m)))
            print(np.shape(fine))
            error=norm(u-u_temp)
            u_temp=u
            k+=1
            print(k)
            print(error)
        return u
    def Parareal_1(self):
        m=self.M
        N=self.N
        course=np.zeros((self.N,self.M))
        fine=np.zeros((self.N,self.M))
        course_temp=np.zeros((self.N,self.M))
        u=np.zeros((self.N,self.M+1))
        u_temp=np.zeros((self.N,self.M+1))
        k=0
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time()
        tol=1e-6
        b_temp=np.empty(shape=(N,1))
        A=np.empty(shape=(N,N))
        B=lambda t: self.stiff.B(t)
        fine_m=8
        d_t=(t[1]-t[0])/8
        M_t=self.mass.Construct()
        F=self.force.Construct()
        M_1_construct=self.mass.Construct_Prob_1()
        M_1_inv=np.linalg.pinv(M_1_construct)
        M_t_inv=np.linalg.inv(M_t)
        M_1=np.empty((m,fine_m,N,N))
        B_t=np.empty((m,fine_m,N,N))
        for i in range(m):
            for j in range(fine_m):
                B_t[i,j,:,:]=B((i*fine_m+j)*d_t)
                if i==0 and k==0:
                    M_1[i,j,:,:]=np.matmul(M_1_inv,d_t*B_t[i,j,:,:])+np.identity(N)
                M_1[i,j,:,:]=np.matmul(M_t_inv,d_t*B_t[i,j,:,:])+np.identity(N)
        M_2=d_t*F
        while error>tol and k<20:
            u[:,0]=u_0
            b_temp=np.matmul((M_1_construct+(1-self.theta)*self.mesh.delta_t()*B(t[0])),u[:,0])+self.mesh.delta_t()*F
            A=(M_t-(self.theta)*self.mesh.delta_t()*B(t[0]))
            x,exit_code=cgs(A=A,b=b_temp, x0=u[:,j])
            if exit_code!=0:
                print("Failed Convergence")
                course[:,0]=np.random.rand(N)
            else:
                if k==0:
                    u[:,1]=x
                course[:,0]=x
            for i in range(2,m+1):
                b_temp=np.matmul((M_t+(1-self.theta)*self.mesh.delta_t()*B(t[j])),u[:,i-1])+self.mesh.delta_t()*F
                A=(M_t-(self.theta)*self.mesh.delta_t()*B(t[i]))
                x,exit_code=cgs(A=A,b=b_temp, x0=u[:,i-1])
                if exit_code!=0:
                    print("Failed Convergence")
                    course[:,i-1]=np.random.rand(N)
                else:
                    if k==0:
                        u[:,i]=x
                    course[:,i-1]=x
                u[:,i]=fine[:,i-1]+course[:,i-1]-course_temp[:,i-1]
                course_temp[:,i-1]=course[:,i-1]
            fine=np.zeros(shape=(self.N,self.M))
            print(np.shape(fine))
            fine=np.transpose(Parallel(n_jobs=-1,verbose=1)\
                (delayed(Fine_Propogator)(M_1[i],M_2,u[:,i],fine[:,i]) for i in range(m)))
            error=norm(u-u_temp)
            print(u)
            u_temp=u
            print(u_temp)
            k+=1
            print(k)
            print(error)
        return u
    

def Fine_Propogator(M_1,M_2,u,u_temp):
    for j in range(8):
        u_temp=np.matmul(M_1[j,:,:],u)+M_2
        u=u_temp
    return u_temp
