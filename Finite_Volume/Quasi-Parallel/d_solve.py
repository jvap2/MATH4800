from ast import arg
from d_mass_matrix import MassMatrix
from d_force_matrix import Force_Matrix
from d_stiff_matrix import StiffMatrix
from d_mesh import Mesh
import cupyx.scipy
from cupyx.scipy.sparse import csc_matrix,linalg
import cupy as cp
from scipy.integrate import quad
import numpy as np
import math
import time
from scipy.linalg import norm
import numba
from numba import cuda
from numba.cuda import stream
import d_cgs
from cupyx.scipy.sparse.linalg import aslinearoperator
from d_cgs import B_1_Cubic_Right_Min,B_2_Cubic_Right_Min,B_3_Cubic_Right_Min, B_1_Cubic_Right_Plus,B_2_Cubic_Right_Plus,B_3_Cubic_Right_Plus
from d_cgs import B_1_Cubic_Left_Min, B_1_Cubic_Left_Plus, B_2_Cubic_Left_Min, B_2_Cubic_Left_Plus, B_3_Cubic_Left_Min, B_3_Cubic_Left_Plus
from d_cgs import K_Min_L,K_Min_R,K_plus_L,K_plus_R





class Final_Solution():
    def __init__(self,a,b,gamma,beta,N,t_0=0,t_m=0,M=0,theta=0):
        self.mass=MassMatrix(a,b,N,t_0, t_m,M)
        self.force=Force_Matrix(a,b,N,t_0,t_m,M)
        self.stiff=StiffMatrix(a,b,N,gamma,beta,t_0,t_m,M)
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.theta=theta
        self.h=self.mesh.silengths()[0]
        self.N=N
        self.gamma=gamma
        self.beta=beta
        self.M=M
        self.mid=self.mesh.midpoints()
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
        u_0=cp.array(self.u_zero_1())
        u=cp.zeros((self.N,self.M+1))
        u[:,0]=u_0
        b=cp.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B_1()),u[:,0])+self.mesh.delta_t()*self.force.Construct()
        A=csc_matrix(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B_1())
        x,exit_code=linalg.cgs(A,b)
        if exit_code !=0:
            print("Failed convergence")
        else:
            u[:,1]=x
        for (i,t) in enumerate(self.mesh.time()[2:]):
            b=cp.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B_1()),u[:,i+1])+self.mesh.delta_t()*self.force.Construct()
            A=csc_matrix(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B_1())
            x,exit_code=linalg.cgs(A,b)
            if exit_code !=0:
                print("Failed convergence")
                break
            else:
                u[:,i+2]=x
        u_return=cp.asnumpy(u)
        print(f"Used bytes before: {mempool.used_bytes()}")
        print(f"Total_bytes before: {mempool.total_bytes()}")
        mempool.free_all_blocks()
        print(f"Used bytes after: {mempool.used_bytes()}")
        print(f"Total_bytes after: {mempool.total_bytes()}")
        return u_return
    def MatInv(self):
        mempool = cp.get_default_memory_pool()
        u_0=self.u_zero_1()
        u=cp.zeros((self.N,self.M+1),dtype=cp.float16)
        u[:,0]=u_0
        F=self.force.Construct_Time()
        for (i,t) in enumerate(self.mesh.time()[1:]):
            if i==0:
                b=cp.matmul((self.mass.Construct_Prob_1()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i], dtype=cp.float16)+self.mesh.delta_t()*F
            else:
                b=cp.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i], dtype=cp.float16)+self.mesh.delta_t()*F
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
    def True_SS_Sol(self):
        sol=lambda x: x-x**2
        u_sol=sol(self.mesh.mesh_points())
        return cp.asnumpy(u_sol)
    def Parareal(self):
        mempool = cp.get_default_memory_pool()
        m=self.M
        N=self.N
        course=cp.zeros((self.N,self.M), dtype=cp.float64)
        course_temp=cp.zeros((self.N,self.M), dtype=cp.float64)
        fine=cp.zeros((self.N,self.M), dtype=cp.float64)
        u=cp.zeros((self.N,self.M+1), dtype=cp.float64)
        u_temp=cp.zeros((self.N,self.M+1), dtype=cp.float64)
        k=0
        error=1
        u_0=self.u_zero(self.mesh.mesh_points()[1:N+1])
        t=self.mesh.time()
        tol=1e-9
        b_temp=cp.empty(shape=(N,1), dtype=cp.float64)
        A=cp.empty(shape=(N,N), dtype=cp.float64)
        B=lambda t: self.stiff.B(t)
        fine_m=10000//m
        d_t=(t[1]-t[0])/fine_m
        M_t=self.mass.Construct()
        F=self.force.Construct()
        M_t_inv=cp.linalg.inv(M_t)
        M_1=cp.empty((m,N,N), dtype=cp.float64)
        B_t=cp.empty((N,N),dtype=cp.float64)
        temp_mat=cp.empty((N,N),dtype=cp.float64)
        for i in range(m):
            for j in range(fine_m):
                B_t=B((i*fine_m+j)*d_t)
                temp_mat=cp.identity(N)+d_t*cp.matmul(M_t_inv,B_t)
                if j==0:
                    M_1[i]=temp_mat
                else:
                    M_1[i]=cp.matmul(temp_mat,M_1[i])
        start=time.time()
        while error>tol:
            u[:,0]=u_0
            for (j,i) in enumerate(range(1,m+1)):
                b_temp=cp.matmul((M_t+(1-self.theta)*self.mesh.delta_t()*B(t[j])),u[:,j])+self.mesh.delta_t()*F
                A=csc_matrix(M_t-(self.theta)*self.mesh.delta_t()*B(t[i]), dtype=cp.float64)
                x,exit_code=linalg.cgs(A=A,b=b_temp, x0=u[:,j], tol=5e-2)
                if exit_code!=0:
                    print("Failed Convergence")
                else:
                    course[:,i-1]=x
                    u[:,i]=fine[:,i-1]+x-course_temp[:,i-1]
                    course_temp[:,i-1]=x
            fine=cp.zeros(shape=(self.N,self.M), dtype=cp.float64)
            threads_per_block=(16,8)
            blocks_per_grid=(((N)+15)//16, (m+7)//8 )
            Fine_Propogator[blocks_per_grid,threads_per_block](M_1,u,fine,m,N)
            print(u)
            dif=u-u_temp
            error=cp.linalg.norm(dif)
            u_temp=u
            k+=1
            print(k)
            print(error)
        end=time.time()
        parareal_time=end-start
        u_return=cp.asnumpy(u)
        mempool.free_all_blocks()
        return u_return, parareal_time
    def Solve_Cubic_1(self):
        mempool = cp.get_default_memory_pool()
        u_0=self.u_zero_1()
        u=cp.zeros((self.N,self.M+1))
        M_1=self.mass.Construct_Prob_1()
        M=self.mass.Construct_Cubic()
        B_L=self.stiff.Cubic_Left_Deriv()
        B_R=self.stiff.Cubic_Right_Test()
        B=B_L+B_R
        F=self.force.Construct_Time()
        u[:,0]=u_0
        start=time.time()
        b=cp.matmul((M_1+(1-self.theta)*self.mesh.delta_t()*B),u[:,0])+self.mesh.delta_t()*F
        A=csc_matrix(M-(self.theta)*self.mesh.delta_t()*B)
        x,exit_code=linalg.cgs(A,b)
        if exit_code !=0:
            print("Failed convergence")
        else:
            u[:,1]=x
        for (i,t) in enumerate(self.mesh.time()[2:]):
            b=cp.matmul((M+(1-self.theta)*self.mesh.delta_t()*B),u[:,i+1])+self.mesh.delta_t()*F
            A=csc_matrix(M-(self.theta)*self.mesh.delta_t()*B)
            x,exit_code=linalg.cgs(A,b)
            if exit_code !=0:
                print("Failed convergence")
                break
            else:
                u[:,i+2]=x
        end=time.time()
        time_total=end-start
        u_return=cp.asnumpy(u)
        print(f"Used bytes before: {mempool.used_bytes()}")
        print(f"Total_bytes before: {mempool.total_bytes()}")
        mempool.free_all_blocks()
        print(f"Used bytes after: {mempool.used_bytes()}")
        print(f"Total_bytes after: {mempool.total_bytes()}")
        return u_return,time_total
    def Steady_State_Cubic_Test(self):
        B=self.stiff.Mem_Effic_B_Left()
        f=self.force.Left_Ex_3()
        u=cp.linalg.solve(-B,f)
        print(u)
        return cp.asnumpy(u)
    def Right_Cubic_Test(self):
        B=self.stiff.Cubic_Right_Test()
        f=self.force.Construct_Right()
        u=cp.linalg.solve(-B,f)
        return cp.asnumpy(u)
    def Lin_Op_Right(self):
        B_1_P=B_1_Cubic_Right_Plus(self.N,self.beta)
        B_2_P=B_2_Cubic_Right_Plus(self.N,self.beta)
        B_3_P=B_3_Cubic_Right_Plus(self.N,self.beta)
        B_1_M=B_1_Cubic_Right_Min(self.N,self.beta)
        B_2_M=B_2_Cubic_Right_Min(self.N,self.beta)
        B_3_M=B_3_Cubic_Right_Min(self.N,self.beta)
        K_p=K_plus_R(self.N,self.h,self.gamma,self.beta,self.mid)
        K_m=K_Min_R(self.N,self.h,self.gamma,self.beta,self.mid)
        b=self.force.Construct_Right()
        r_0=cp.ones(shape=self.N)
        x=cp.zeros(shape=self.N)
        A=K_p.matvec(B_1_P.matvec(x[:2])+B_2_P.matvec(x[2:self.N-3])+B_3_P.matvec(x[self.N-3:]))-K_m.matvec(B_1_M.matvec(x[:3])+B_2_M.matvec(x[2:self.N-3])+B_3_M.matvec(x[self.N-3:]))
        x,exit_code=linalg.cgs(A,b)
        if exit_code!=0:
            print("Unable to converge")
        # p,u=r,r
        # norm=5
        # while norm>=1e-6:
        #     alpha=cp.dot(r,r_0)/cp.dot(-K_p.matvec(B_1_P.matvec(p[:2])+B_2_P.matvec(p[2:self.N-3])+B_3_P.matvec(p[self.N-3:]))-K_m.matvec(B_1_M.matvec(p[:3])+B_2_M.matvec(p[2:self.N-3])+B_3_M.matvec(p[self.N-3:])),r_0)
        #     q=u-alpha*(-K_p.matvec(B_1_P.matvec(p[:2])+B_2_P.matvec(p[2:self.N-3])+B_3_P.matvec(p[self.N-3:]))-K_m.matvec(B_1_M.matvec(p[:3])+B_2_M.matvec(p[2:self.N-3])+B_3_M.matvec(p[self.N-3:])))
        #     x_new=x+alpha*(u+q)
        #     r_new=r-alpha*(-K_p.matvec(B_1_P.matvec(u[:2]+q[:2])+B_2_P.matvec(u[2:self.N-3]+q[2:self.N-3])+B_3_P.matvec(u[self.N-3:]+q[self.N-3:]))-K_m.matvec(B_1_M.matvec(u[:3]+q[:3])+B_2_M.matvec(u[2:self.N-3]+q[2:self.N-3])+B_3_M.matvec(u[self.N-3:]+q[self.N-3:])))
        #     Beta=cp.dot(r_new,r_0)/cp.dot(r,r_0)
        #     u=r_new+Beta*q
        #     p_new=u+Beta*(q+Beta*p)
        #     p=p_new
        #     r=r_new
        #     norm=abs(cp.max(x_new-x))
        #     x=x_new
        x=cp.asnumpy(x)
        return x
    def Lin_Op_Left(self):
        B_1_P=B_1_Cubic_Left_Plus(self.N,self.beta)
        B_2_P=B_2_Cubic_Left_Plus(self.N,self.beta)
        B_3_P=B_3_Cubic_Left_Plus(self.N,self.beta)
        B_1_M=B_1_Cubic_Left_Min(self.N,self.beta)
        B_2_M=B_2_Cubic_Left_Min(self.N,self.beta)
        B_3_M=B_3_Cubic_Left_Min(self.N,self.beta)
        K_p=K_plus_L(self.N,self.h,self.gamma,self.beta,self.mid)
        K_m=K_Min_L(self.N,self.h,self.gamma,self.beta,self.mid)
        b=self.force.Left_Ex_1()
        r_0=cp.ones(shape=self.N)
        x=cp.zeros(shape=self.N)
        A=K_p.matvec(B_1_P.matvec(x[:3])+B_2_P.matvec(x[3:self.N-2])+B_3_P.matvec(x[self.N-3:]))-K_m.matvec(B_1_M.matvec(x[:3])+B_2_M.matvec(x[3:self.N-2])+B_3_M.matvec(x[self.N-2:]))
        x,exit_code=linalg.cgs(A,b)
        if exit_code!=0:
            print("Unable to converge")
        # norm=5
        # while norm>=1e-8:
        #     alpha=cp.dot(r,r_0)/cp.dot(-K_p.matvec(B_1_P.matvec(p[:3])+B_2_P.matvec(p[3:self.N-2])+B_3_P.matvec(p[self.N-3:]))-K_m.matvec(B_1_M.matvec(p[:3])+B_2_M.matvec(p[3:self.N-2])+B_3_M.matvec(p[self.N-2:])),r_0)
        #     q=u-alpha*(-K_p.matvec(B_1_P.matvec(p[:3])+B_2_P.matvec(p[3:self.N-2])+B_3_P.matvec(p[self.N-3:]))-K_m.matvec(B_1_M.matvec(p[:3])+B_2_M.matvec(p[3:self.N-2])+B_3_M.matvec(p[self.N-2:])))
        #     x_new=x+alpha*(u+q)
        #     r_new=r-alpha*(-K_p.matvec(B_1_P.matvec(u[:3]+q[:3])+B_2_P.matvec(u[3:self.N-2]+q[3:self.N-2])+B_3_P.matvec(u[self.N-3:]+q[self.N-3:]))-K_m.matvec(B_1_M.matvec(u[:3]+q[:3])+B_2_M.matvec(u[3:self.N-2]+q[3:self.N-2])+B_3_M.matvec(u[self.N-2:]+q[self.N-2:])))
        #     Beta=cp.dot(r_new,r_0)/cp.dot(r,r_0)
        #     u=r_new+Beta*q
        #     p_new=u+Beta*(q+Beta*p)
        #     p=p_new
        #     r=r_new
        #     norm=abs(cp.max(x_new-x))
        #     x=x_new
        x=cp.asnumpy(x)
        return x
    def Test_CGS(self):
        Stiff=self.stiff.Cubic_Right_Test()
        B=aslinearoperator(-Stiff)
        b=self.force.Construct_Right()
        r_0=cp.ones(shape=self.N)
        x=cp.zeros(shape=self.N)
        r=b-B.matvec(x)
        p,u=r,r
        norm=5
        while norm>=1e-6:
            alpha=cp.dot(r,r_0)/cp.dot(B.matvec(p),r_0)
            q=u-alpha*(B.matvec(p))
            x_new=x+alpha*(u+q)
            r_new=r-alpha*(B.matvec(u+q))
            Beta=cp.dot(r_new,r_0)/cp.dot(r,r_0)
            u=r_new+Beta*q
            p_new=u+Beta*(q+Beta*p)
            p=p_new
            r=r_new
            norm=abs(cp.max(x_new-x))
            x=x_new
        x=cp.asnumpy(x)
        return x
    def Steady_State_Linear(self):
        f=self.force.Left_Ex_3()
        B=self.stiff.Linear_Left_Deriv()
        u=cp.linalg.solve(-B,f)
        return cp.asnumpy(u)
    def Linear_Right(self):
        f=self.force.Construct_Right()
        B=self.stiff.Linear_Right_Deriv()
        u=cp.linalg.solve(-B,f)
        return cp.asnumpy(u)
    def Parareal_1(self):
        mempool = cp.get_default_memory_pool()
        m=self.M
        N=self.N
        course=cp.zeros((self.N,self.M), dtype=cp.float64)
        fine=cp.zeros((self.N,self.M), dtype=cp.float64)
        course_temp=cp.zeros((self.N,self.M), dtype=cp.float64)
        u=cp.zeros((self.N,self.M+1), dtype=cp.float64)
        u_temp=cp.zeros((self.N,self.M+1), dtype=cp.float64)
        k=0
        error=1
        u_0=self.u_zero_1()
        t=self.mesh.time()
        tol=1e-9
        b_temp=cp.empty(shape=(N,1), dtype=cp.float64)
        A=cp.empty(shape=(N,N), dtype=cp.float64)
        B=self.stiff.B_1()
        fine_m=10000//m
        d_t=(t[1]-t[0])/fine_m
        M_t=self.mass.Construct_Lump()
        F=self.force.Construct()
        M_1_construct=self.mass.Construct_Prob_1()
        M_t_inv=cp.linalg.inv(M_t)
        M_1=cp.empty(shape=(m,N,N))
        Mat_for_fine_1=cp.matmul(M_t_inv,d_t*B)+cp.matmul(M_t_inv,M_1_construct)
        Mat_for_fine=cp.matmul(M_t_inv,d_t*B)+cp.identity(N)
        for j in range(m):
            if j==0:
                M_1[j]=cp.matmul(cp.linalg.matrix_power(Mat_for_fine,fine_m-1),Mat_for_fine_1)
            else:
                M_1[j]=cp.linalg.matrix_power(Mat_for_fine,fine_m)
        start=time.time()
        while error>tol:
            u[:,0]=u_0
            b_temp=cp.matmul((M_1_construct+(1-self.theta)*self.mesh.delta_t()*B),u[:,0])+self.mesh.delta_t()*F
            A=csc_matrix(M_t-(self.theta)*self.mesh.delta_t()*B, dtype=cp.float64)
            x,exit_code=linalg.gmres(A=A,b=b_temp,tol=5e-2)
            if exit_code!=0:
                print("Failed Convergence")
            else:
                course[:,0]=x
                u[:,1]=fine[:,0]+x-course_temp[:,0]
                course_temp[:,0]=x
            for i in range(2,m+1):
                b_temp=cp.matmul((M_t+(1-self.theta)*self.mesh.delta_t()*B),u[:,i-1])+self.mesh.delta_t()*F
                A=csc_matrix(M_t-(self.theta)*self.mesh.delta_t()*B)
                x,exit_code=linalg.gmres(A=A,b=b_temp, tol=5e-2)
                if exit_code!=0:
                    print("Failed Convergence")
                else:
                    course[:,i-1]=x
                    u[:,i]=fine[:,i-1]+x-course_temp[:,i-1]
                    course_temp[:,i-1]=x
            fine=cp.zeros(shape=(self.N,self.M))
            threads_per_block=(16,8)
            blocks_per_grid=(((N)+16-1)//16, (m+7)//8 )
            Fine_Propogator[blocks_per_grid,threads_per_block](M_1,u,fine,m,N)
            print(u)
            dif=u-u_temp
            error=cp.linalg.norm(dif)
            u_temp=u
            k+=1
            print(k)
            print(error)
        end=time.time()
        parareal_time=end-start
        u_return=cp.asnumpy(u)
        mempool.free_all_blocks()
        return u_return,parareal_time
    def MatInv_Cubic(self):
        mempool = cp.get_default_memory_pool()
        u_0=self.u_zero_1()
        u=cp.zeros((self.N,self.M+1))
        M_1=self.mass.Construct_Prob_1()
        M=self.mass.Construct_Cubic()
        B_L=self.stiff.Cubic_Left_Deriv()
        B_R=self.stiff.Cubic_Right_Test()
        B=B_L+B_R
        F=self.force.Construct_Time()
        u[:,0]=u_0
        start=time.time()
        b=cp.matmul((M_1+(1-self.theta)*self.mesh.delta_t()*B),u[:,0])+self.mesh.delta_t()*F
        A=(M-(self.theta)*self.mesh.delta_t()*B)
        u[:,1]=cp.matmul(cp.linalg.inv(A),b)
        for (i,t) in enumerate(self.mesh.time()[2:]):
            b=cp.matmul((M+(1-self.theta)*self.mesh.delta_t()*B),u[:,i+1])+self.mesh.delta_t()*F
            A=(M-(self.theta)*self.mesh.delta_t()*B)
            u[:,i+2]=cp.matmul(cp.linalg.inv(A),b)
        end=time.time()
        time_total=end-start
        u_return=cp.asnumpy(u)
        print(f"Used bytes before: {mempool.used_bytes()}")
        print(f"Total_bytes before: {mempool.total_bytes()}")
        mempool.free_all_blocks()
        print(f"Used bytes after: {mempool.used_bytes()}")
        print(f"Total_bytes after: {mempool.total_bytes()}")
        return u_return
    
@cuda.jit
def Fine_Propogator(M,u,u_fine,m,N):
    row,col=cuda.grid(2)
    fSum=0
    if(row<N and col<m):
        for k in range(N):
            fSum+=M[col,row,k]*u[k,col]
        u_fine[row,col]=fSum

