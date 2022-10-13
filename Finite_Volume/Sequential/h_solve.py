from re import M
from unittest import findTestCases
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
    def CGS_1(self):
        u_0=self.u_zero_1()
        u=np.zeros((self.N,self.M+1))
        u[:,0]=u_0
        for (i,t) in enumerate(self.mesh.time()[1:]):
            if i==0:
                b=np.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B_1()),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            else:
                b=np.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B_1()),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            A=csc_matrix(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B_1())
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
        course=np.zeros((self.N,self.M), dtype=np.float32)
        fine=np.zeros((self.N,self.M), dtype=np.float32)
        course_temp=np.zeros((self.N,self.M), dtype=np.float32)
        u=np.zeros((self.N,self.M+1), dtype=np.float32)
        u_temp=np.zeros((self.N,self.M+1), dtype=np.float32)
        u_fine=np.zeros((self.N,self.M+1), dtype=np.float32)
        k=0
        error=1
        u_0=self.u_zero(self.mesh.mesh_points()[1:N+1])
        t=self.mesh.time()
        tol=1e-6
        b_temp=np.empty(shape=(N,1), dtype=np.float32)
        A=np.empty(shape=(N,N), dtype=np.float32)
        B=lambda t: self.stiff.B(t)
        fine_m=4096//m
        d_t=(t[1]-t[0])/fine_m
        M_t=self.mass.Construct_Lump()
        F=self.force.Construct()
        M_t_inv=np.linalg.inv(M_t)
        M_1=np.empty((m,fine_m,N,N), dtype=np.float32)
        B_t=np.empty((m,fine_m,N,N), dtype=np.float32)
        for i in range(m):
            for j in range(fine_m):
                B_t[i,j,:,:]=B((i*fine_m+j)*d_t)
                M_1[i,j,:,:]=np.matmul(M_t_inv,d_t*B_t[i,j,:,:])+np.identity(N)
        print(np.shape(M_1))
        M_2=d_t*F
        while error>tol or k<20:
            u[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                b_temp=np.matmul((M_t+(1-self.theta)*self.mesh.delta_t()*B(t[j])),u[:,j])+self.mesh.delta_t()*F
                A=csc_matrix(M_t-(self.theta)*self.mesh.delta_t()*B(t[i]), dtype=np.float32)
                x,exit_code=cgs(A=A,b=b_temp, x0=u[:,j])
                if exit_code!=0:
                    print("Failed Convergence")
                    course[:,j]=np.random.rand(N)
                else:
                    course[:,j]=x
                u[:,i]=fine[:,j]+course[:,j]-course_temp[:,j]
            course_temp=course
            u_fine=u
            fine=np.zeros(shape=(self.N,self.M))
            print(np.shape(fine))
            fine=np.transpose(Parallel(n_jobs=-1,verbose=1)\
                (delayed(Fine_Propogator)(M_1[i],M_2,u_fine[:,i],fine[:,i],fine_m) for i in range(m)))
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
        course=np.zeros((self.N,self.M), dtype=np.float32)
        fine=np.zeros((self.N,self.M), dtype=np.float32)
        course_temp=np.zeros((self.N,self.M), dtype=np.float32)
        u=np.zeros((self.N,self.M+1), dtype=np.float32)
        u_temp=np.zeros((self.N,self.M+1), dtype=np.float32)
        k=0
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time()
        tol=1e-9
        b_temp=np.empty(shape=(N,1), dtype=np.float32)
        A=np.empty(shape=(N,N), dtype=np.float32)
        B=self.stiff.B_1()
        fine_m=10000//m
        d_t=(t[1]-t[0])/fine_m
        M_t=self.mass.Construct()
        M_Lump=self.mass.Construct_Lump()
        F=self.force.Construct()
        M_1_construct=self.mass.Construct_Prob_1()
        M_t_inv=np.linalg.inv(M_t)
        M_1=np.empty(shape=(m,N,N))
        Mat_for_fine_1=np.matmul(M_t_inv,d_t*B)+np.matmul(M_t_inv,M_1_construct)
        Mat_for_fine=np.matmul(M_t_inv,d_t*B)+np.identity(N)
        for j in range(m):
            if j==0:
                M_1[j]=np.matmul(np.linalg.matrix_power(Mat_for_fine,fine_m-1),Mat_for_fine_1)
            else:
                M_1[j]=np.linalg.matrix_power(Mat_for_fine,fine_m)
        while error>tol:
            u[:,0]=u_0
            b_temp=np.matmul((M_1_construct+(1-self.theta)*self.mesh.delta_t()*B),u[:,0])+self.mesh.delta_t()*F
            A=csc_matrix(M_Lump-(self.theta)*self.mesh.delta_t()*B, dtype=np.float32)
            x,exit_code=gmres(A=A,b=b_temp,tol=5e-3)
            if exit_code!=0:
                print("Failed Convergence")
            else:
                course[:,0]=x
                u[:,1]=fine[:,0]+x-course_temp[:,0]
                course_temp[:,0]=x
            for i in range(2,m+1):
                b_temp=np.matmul((M_Lump+(1-self.theta)*self.mesh.delta_t()*B),u[:,i-1])+self.mesh.delta_t()*F
                A=csc_matrix(M_Lump-(self.theta)*self.mesh.delta_t()*B)
                x,exit_code=gmres(A=A,b=b_temp, tol=5e-3)
                if exit_code!=0:
                    print("Failed Convergence")
                else:
                    course[:,i-1]=x
                    u[:,i]=fine[:,i-1]+x-course_temp[:,i-1]
                    course_temp[:,i-1]=x
            print('U:\n',u)
            print('U_temp:\n',u_temp)
            print('fine:\n',fine)
            error=np.linalg.norm(u-u_temp)
            u_temp=u
            fine=np.zeros(shape=(self.N,self.M), dtype=np.float32)
            fine=np.transpose(Parallel(n_jobs=-1,verbose=1,backend='threading')\
                (delayed(Fine_Propogator)(M_1[i],u[:,i],fine[:,i]) for i in range(m)))
            k+=1
            print(k)
            print(error)
        return u
    

def Fine_Propogator(Mat_1,u_fine,u_tmp):
    u_tmp=np.matmul(Mat_1,u_fine)
    return u_tmp

def Error(v_1,v_2):
    err=-1
    error_temp=0
    for i in range(np.shape(v_1)[0]):
        for j in range(np.shape(v_1)[1]):
            error_temp=abs(v_1[i,j]-v_2[i,j])
        if (error_temp>err):
            err=error_temp
    return err


