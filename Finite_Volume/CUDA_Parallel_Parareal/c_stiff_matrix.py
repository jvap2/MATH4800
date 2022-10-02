from concurrent.futures import thread
import numpy as np
import numba
from numba import cuda
from c_mesh import Mesh
from c_mesh import p_linspace, init_threads_blocks, helper_linspace
from scipy.linalg import toeplitz
import math
from math import gamma as g


class StiffMatrix():
    def __init__(self,a,b,N,t_0,t_m,M,gamma,beta):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.gamma=gamma
        self.beta=beta
        self.N=self.mesh.NumofSubIntervals()
        self.x=self.mesh.mesh_points()
        self.mid=self.mesh.midpoints()
        self.h=self.mesh.silengths()
    def BL(self, t=0):
        Nsize=self.N
        Nsize_2=self.N-2
        col_linspace=np.empty(shape=Nsize_2)
        row_linspace=np.empty(shape=Nsize_2)
        col,row=np.zeros(shape=(Nsize)),np.zeros(shape=(Nsize))
        d_col,d_row=cuda.to_device(col),cuda.to_device(row)
        d_c_lin,d_r_lin=cuda.to_device(col_linspace),cuda.to_device(row_linspace)
        thread_lin, block_lin = init_threads_blocks(16,Nsize_2)
        p_linspace[block_lin,thread_lin](d_c_lin,-2,1-Nsize,Nsize_2)
        p_linspace[block_lin,thread_lin](d_r_lin,2,Nsize-1,Nsize_2)
        p_init_BL[block_lin,thread_lin](d_row,d_col,d_r_lin,d_c_lin,self.beta,self.gamma, Nsize_2)
        d_col.copy_to_host(col)
        d_row.copy_to_host(row)
        col[0],row[0]=(1-self.gamma)*(3/2)**self.beta-(2-self.gamma)*(1/2)**self.beta,(1-self.gamma)*(3/2)**self.beta-(2-self.gamma)*(1/2)**self.beta
        col[1],row[1]=(1-self.gamma)*((1/2)**self.beta+(5/2)**self.beta-2*(3/2)**self.beta),(1+self.gamma)*(1/2)**self.beta-(self.gamma)*(3/2)**self.beta
        BL=toeplitz(c=col,r=row)
        return BL
    def BR(self, t=0):
        Nsize=self.N
        Nsize_2=self.N-2
        beta=self.beta
        gamma=self.gamma
        col_linspace=np.empty(shape=Nsize_2)
        row_linspace=np.empty(shape=Nsize_2)
        col,row=np.zeros(shape=(Nsize)),np.zeros(shape=(Nsize))
        d_col,d_row=cuda.to_device(col),cuda.to_device(row)
        d_c_lin,d_r_lin=cuda.to_device(col_linspace),cuda.to_device(row_linspace)
        thread_lin, block_lin = init_threads_blocks(16,Nsize_2)
        p_linspace[block_lin,thread_lin](d_c_lin,-2,1-Nsize,Nsize_2)
        p_linspace[block_lin,thread_lin](d_r_lin,2,Nsize-1,Nsize_2)
        p_init_BR[block_lin,thread_lin](d_row,d_col,d_r_lin,d_c_lin,beta,gamma,Nsize_2)
        d_col.copy_to_host(col)
        d_row.copy_to_host(row)
        col[0],row[0]=self.gamma*(3/2)**self.beta-(1+self.gamma)*(1/2)**self.beta,self.gamma*(3/2)**self.beta-(1+self.gamma)*(1/2)**self.beta
        col[1],row[1]=(2-self.gamma)*(1/2)**self.beta-(1-self.gamma)*(3/2)**self.beta,self.gamma*((5/2)**self.beta-2*(3/2)**self.beta+(1/2)**self.beta)
        BR=toeplitz(c=col, r=row)
        return BR
    def B(self, t=0):
        Nsize=self.N
        h=self.h
        beta=self.beta
        K_m_1=.01*np.ones(Nsize)
        threads_per_block=128
        blocks_per_grid=(Nsize+threads_per_block-1)//threads_per_block
        threads_per_block_2D=(128,128)
        blocks_per_grid_2D=((Nsize+threads_per_block_2D[0]-1)//threads_per_block_2D[0],(Nsize+threads_per_block_2D[1]-1)//threads_per_block_2D[1])
        B=np.empty(shape=(Nsize,Nsize))
        K=np.diag(K_m_1,k=0)
        print(K)
        B_L_res,B_R_res=np.empty(shape=(Nsize,Nsize)),np.empty(shape=(Nsize,Nsize))
        d_BLres,d_BRres=cuda.to_device(B_L_res),cuda.to_device(B_R_res)
        d_B=cuda.to_device(B)
        d_K=cuda.to_device(K)
        B_L,B_R=self.BL(),self.BR()
        d_B_L,d_B_R=cuda.to_device(B_L),cuda.to_device(B_R)
        d_h=cuda.to_device(h)
        p_mat_mult[blocks_per_grid_2D,threads_per_block_2D](d_K,d_B_L,d_BLres,Nsize)
        p_mat_mult[blocks_per_grid_2D,threads_per_block_2D](d_K,d_B_R,d_BRres,Nsize)
        const=1/g(1+beta)
        p_assemble_B[blocks_per_grid_2D,threads_per_block_2D](d_B,d_B_L,d_B_R,d_h,const,beta,Nsize)
        B=np.empty(shape=(Nsize,Nsize))
        d_B.copy_to_host(B)
        return B
    def B_P_2(self,t):
        Nsize=self.N
        h=self.h
        beta=self.beta
        x_arr_1=self.mesh.midpoints()[:Nsize]
        x_arr_2=self.mesh.midpoints()[1:]
        d_x_arr_1=cuda.to_device(x_arr_1)
        d_x_arr_2=cuda.to_device(x_arr_2)
        k_arr_1=np.empty(Nsize)
        k_arr_2=np.empty(Nsize)
        d_k_arr_1=cuda.to_device(k_arr_1)
        d_k_arr_2=cuda.to_device(k_arr_2)
        threads_per_block=16
        blocks_per_grid=(Nsize+threads_per_block-1)//threads_per_block
        threads_per_block_2D=(128,128)
        blocks_per_grid_2D=((Nsize+threads_per_block_2D[0]-1)//threads_per_block_2D[0],(Nsize+threads_per_block_2D[1]-1)//threads_per_block_2D[1])
        B=np.empty(shape=(Nsize,Nsize))
        K_1=np.empty(shape=(Nsize,Nsize))
        K_2=np.empty(shape=(Nsize,Nsize))
        B_L_res,B_R_res=np.empty(shape=(Nsize,Nsize)),np.empty(shape=(Nsize,Nsize))
        d_BLres,d_BRres=cuda.to_device(B_L_res),cuda.to_device(B_R_res)
        d_B=cuda.to_device(B)
        d_K_1=cuda.to_device(K_1)
        d_K_2=cuda.to_device(K_2)
        B_L,B_R=self.BL(),self.BR()
        d_B_L,d_B_R=cuda.to_device(B_L),cuda.to_device(B_R)
        d_h=cuda.to_device(h)
        non_const_diff[blocks_per_grid,threads_per_block](d_k_arr_1,t,d_x_arr_1,Nsize)
        non_const_diff[blocks_per_grid,threads_per_block](d_k_arr_2,t,d_x_arr_2,Nsize)
        diag_K_P2[blocks_per_grid,threads_per_block](d_K_1,d_k_arr_1,Nsize)
        diag_K_P2[blocks_per_grid,threads_per_block](d_K_2,d_k_arr_2,Nsize)
        p_mat_mult[blocks_per_grid_2D,threads_per_block_2D](d_K_1,d_B_L,d_BLres,Nsize)
        p_mat_mult[blocks_per_grid_2D,threads_per_block_2D](d_K_2,d_B_R,d_BRres,Nsize)
        const=1/g(1+beta)
        p_assemble_B[blocks_per_grid_2D,threads_per_block_2D](d_B,d_B_L,d_B_R,d_h,const,beta,Nsize)
        B=np.empty(shape=(Nsize,Nsize))
        d_B.copy_to_host(B)
        return B


@cuda.jit("void(float64[:], float64[:], float64[:], float64[:], float64,float64,float64)")
def p_init_BL(row_array,col_array, r_lin, c_lin, beta,gamma,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        col_array[2:]=(1-gamma)*((-c_lin[idx]-(1/2))**beta+(-c_lin[idx]+(3/2))**beta-2*(-c_lin[idx]+(1/2))**beta)
        row_array[2:]=gamma*(2*(r_lin[idx]-(1/2))**beta-(r_lin[idx]+(1/2))**beta-(r_lin[idx]-(3/2))**beta)

@cuda.jit("void(float64[:], float64[:], float64[:], float64[:], float64,float64,float64)")
def p_init_BR(row_array,col_array, r_lin, c_lin, beta,gamma,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        col_array[idx+2]=(1-gamma)*(2*(-c_lin[idx]-(1/2))**beta-(-c_lin[idx]-(3/2))**beta-(-c_lin[idx]+(1/2))**beta)
        row_array[idx+2]=gamma*((r_lin[idx]+(3/2))**beta-2*(r_lin[idx]+(1/2))**beta+(r_lin[idx]-(1/2))**beta)

@cuda.jit("void(float64[:,:],float64,float64)")
def p_diag_K_1(mat,K,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        mat[idx,idx]=K

@cuda.jit("void(float64[:,:],float64[:,:],float64[:,:],int32)")
def p_mat_mult(A,B,C,Nsize):
    col=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    row=cuda.threadIdx.y+(cuda.blockDim.y*cuda.blockIdx.y)
    fSum=0
    for i in range(Nsize):
        fSum+=A[row,i]*B[i,col]
    C[row,col]=fSum


@cuda.jit("void(float64[:,:],float64[:,:],float64[:,:],float64[:],float64,float64,float64)")
def p_assemble_B(B,BL,BR,h,const,beta,Nsize):
    col=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    row=cuda.threadIdx.y+(cuda.blockDim.y*cuda.blockIdx.y)
    if row<Nsize and col<Nsize:
        B[row,col]=const*((1/(h[col]**(1-beta)))*(BL[row,col])+(1/(h[col+1]**(1-beta)))*(BR[row,col]))

@cuda.jit("void(float64[:,:],float64[:],float64)")
def diag_K_P2(mat,array,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if idx<Nsize:
        mat[idx,idx]=array[idx]


@cuda.jit("void(float64[:],float64,float64[:],float64)")
def non_const_diff(arr,t,x_arr,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if idx<Nsize:
        arr[idx]=.002*(1+x_arr[idx]*(2-x_arr[idx])+t**2)



b=StiffMatrix(0,2,4,0,1,4,.5,.5)
print(b.B_P_2(t=0))