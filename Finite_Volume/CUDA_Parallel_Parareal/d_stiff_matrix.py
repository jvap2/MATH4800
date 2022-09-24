import numpy as np
import numba
from numba import cuda
from c_mesh import Mesh
from c_mesh import p_linspace, init_threads_blocks
from scipy.linalg import toeplitz
import math


class StiffMatrix():
    def __init__(self,a,b,N,t_0,t_m,M,gamma,beta):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.gamma=gamma
        self.beta=beta
        self.N=self.mesh.NumofSubIntervals()
        self.x=self.mesh.mesh_points()
        self.mid=self.mesh.midpoints()
        self.h=self.mesh.silengths()
    def BL(self):
        Nsize=self.N
        Nsize_2=self.N-2
        col_linspace=np.empty(shape=Nsize_2)
        row_linspace=np.empty(shape=Nsize_2)
        col,row=np.empty(shape=(Nsize)),np.empty(shape=(Nsize))
        d_col,d_row=cuda.to_device(col),cuda.to_device(row)
        d_c_lin,d_r_lin=cuda.to_device(col_linspace),cuda.to_device(row_linspace)
        thread_lin, block_lin = init_threads_blocks(128,Nsize_2)
        p_linspace[block_lin,thread_lin](d_c_lin,-2,1-Nsize,Nsize_2)
        p_linspace[block_lin,thread_lin](d_r_lin,2,Nsize-1,Nsize_2)
        p_init_BL[block_lin,thread_lin](d_row,d_col,d_r_lin,d_c_lin,self.beta,self.gamma, Nsize_2)
        d_col.copy_to_host(col)
        d_row.copy_to_host(row)
        col[0],row[0]=(1-self.gamma)*(3/2)**self.beta-(2-self.gamma)*(1/2)**self.beta,(1-self.gamma)*(3/2)**self.beta-(2-self.gamma)*(1/2)**self.beta
        col[1],row[1]=(1-self.gamma)*((1/2)**self.beta+(5/2)**self.beta-2*(3/2)**self.beta),(1+self.gamma)*(1/2)**self.beta-(self.gamma)*(3/2)**self.beta
        BL=toeplitz(c=col,r=row)
        return BL
    def BR(self):
        Nsize=self.N
        Nsize_2=self.N-2
        col_linspace=np.empty(shape=Nsize_2)
        row_linspace=np.empty(shape=Nsize_2)
        col,row=np.empty(shape=(Nsize)),np.empty(shape=(Nsize))
        d_col,d_row=cuda.to_device(col),cuda.to_device(row)
        d_c_lin,d_r_lin=cuda.to_device(col_linspace),cuda.to_device(row_linspace)
        thread_lin, block_lin = init_threads_blocks(128,Nsize_2)
        p_linspace[block_lin,thread_lin](d_c_lin,-2,1-Nsize,Nsize_2)
        p_linspace[block_lin,thread_lin](d_r_lin,2,Nsize-1,Nsize_2)
        p_init_BR[block_lin,thread_lin](d_row,d_col,d_r_lin,d_c_lin,self.beta,self.gamma, Nsize_2)
        d_col.copy_to_host(col)
        d_row.copy_to_host(row)
        col[0],row[0]=self.gamma*(3/2)**self.beta-(1+self.gamma)*(1/2)**self.beta,self.gamma*(3/2)**self.beta-(1+self.gamma)*(1/2)**self.beta
        col[1],row[1]=(2-self.gamma)*(1/2)**self.beta-(1-self.gamma)*(3/2)**self.beta,self.gamma*((5/2)**self.beta-2*(3/2)**self.beta+(1/2)**self.beta)
        BR=toeplitz(c=col, r=row)
        return BR
    def B(self):
        Nsize=self.N
        h=self.h
        beta=self.beta
        K_m_1=.01
        thread_lin_y, block_lin_y = init_threads_blocks(128,Nsize)
        thread_lin_x, block_lin_x = init_threads_blocks(128,Nsize)
        thread=(thread_lin_x,thread_lin_y)
        block=(block_lin_x,block_lin_y)
        B=np.empty(shape=(Nsize,Nsize))
        d_B=cuda.to_device(B)
        B_L,B_R=self.BL(),self.BR()
        d_B_L,d_B_R=cuda.to_device(B_L),cuda.to_device(B_R)
        p_assemble_B[block,thread](d_B,d_B_L,d_B_R,h,K_m_1,beta,Nsize)
        d_B.copy_to_host(B)
        return B


@cuda.jit("void(float64[:], float64[:], float64[:], float64[:], float64,float64,float64)")
def p_init_BL(row_array,col_array, r_lin, c_lin, beta,gamma,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        row_array[idx+2]=(1-gamma)*(2*(-r_lin[idx]-.5)**beta-(-r_lin[idx]-1.5)**beta-(-r_lin[idx]+.5)**beta)
        col_array[idx+2]=(gamma)*((c_lin[idx]+1.5)**beta-2*(c_lin[idx]+.5)**beta+(c_lin[idx]-1.5)**beta)

@cuda.jit("void(float64[:], float64[:], float64[:], float64[:], float64,float64,float64)")
def p_init_BR(row_array,col_array, r_lin, c_lin, beta,gamma,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        row_array[idx+2]=(1-gamma)*((-r_lin[idx]-.5)**beta+(-r_lin[idx]+1.5)**beta-2*(-r_lin[idx]+.5)**beta)
        col_array[idx+2]=gamma*(2*(c_lin[idx]-.5)**beta-(c_lin[idx]+.5)**beta-(c_lin[idx]-1.5)**beta)

@cuda.jit("void(float64[:,:],float64[:,:],float[:,:],float64[:],float64,float64,float64)")
def p_assemble_B(B,BL,BR,h,K,beta,Nsize):
    col=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    row=cuda.threadIdx.y+(cuda.blockDim.y*cuda.blockIdx.y)
    if(row<Nsize & col<Nsize):
        B[row,col]=K/(math.gamma(1+beta))*((1/h[col])*(BL[row,col])+(1/h[col+1])*(BR[row,col]))
