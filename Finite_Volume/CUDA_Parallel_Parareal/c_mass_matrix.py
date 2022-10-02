import numpy as np
import numba
from numba import cuda
from c_mesh import Mesh

class MassMatrix():
    def __init__(self, a, b, N, t_0, t_m, M):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=N
    def Construct(self):
        Nsize_mid=self.N
        Nsize_other=self.N-1
        x=self.mesh.mesh_points()
        mid=self.mesh.midpoints()
        h=self.mesh.silengths()
        Mass=np.zeros((self.N,self.N))
        d_Mass=cuda.to_device(Mass)
        d_x,d_midpoints,d_h=cuda.to_device(x),cuda.to_device(mid),cuda.to_device(h)
        threads_per_block=128
        blocks_per_grid=(self.N+threads_per_block-1)//threads_per_block
        p_middle_diag[blocks_per_grid,threads_per_block](d_Mass,d_h,d_midpoints,d_x,Nsize_mid)
        p_upper_diag[blocks_per_grid,threads_per_block](d_Mass,d_h,d_midpoints,d_x,Nsize_other)
        p_lower_diag[blocks_per_grid,threads_per_block](d_Mass,d_h,d_midpoints,d_x,Nsize_other)
        d_Mass.copy_to_host(Mass)
        return Mass
    def Construct_Prob_1_Init(self):
        M=np.zeros((self.N))
        M[(self.N)//2]=1
        M=np.diag(M,k=0)
        return M


@cuda.jit("void(float64[:,:],float64[:],float64[:],float64[:],float64)")
def p_middle_diag(array,h,mid,x,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        array[idx,idx]=(1/(2*h[idx]))*((x[idx+1]**2)-2*x[idx]*(x[idx+1]-mid[idx])-mid[idx]**2)
        array[idx,idx]+=(1/(2*h[idx+1]))*((x[idx+1]**2)-2*x[idx+2]*(x[idx+1]-mid[idx+1])-mid[idx+1]**2)

@cuda.jit("void(float64[:,:],float64[:],float64[:],float64[:],float64)")
def p_lower_diag(array,h,mid,x,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        array[idx+1,idx]=(1/(2*h[idx+1]))*(x[idx+2]**2-2*x[idx+2]*mid[idx+1]+mid[idx+1]**2)

@cuda.jit("void(float64[:,:],float64[:],float64[:],float64[:],float64)")
def p_upper_diag(array,h,mid,x,Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        array[idx,idx+1]=(1/(2*h[idx]))*((mid[idx]**2)-2*x[idx]*mid[idx]+x[idx]**2)
