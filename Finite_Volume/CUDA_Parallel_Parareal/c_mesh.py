from array import array
from concurrent.futures import thread
from math import floor
import numpy as np
import numba
from numba import cuda



class Mesh():
    def __init__(self,a,b,N,t_0,t_m,M):
        self.a=a
        self.b=b
        self.N=N
        self.t_0=t_0
        self.t_m=t_m
        self.M=M
    def NumofSubIntervals(self):
        return self.N
    def mesh_points(self):
        b,a,Nsize=self.b,self.a,self.N+2
        array=np.zeros(Nsize)
        threads_per_block, blocks_per_grid= init_threads_blocks(128, Nsize)
        d_array=cuda.to_device(array)
        p_linspace[blocks_per_grid,threads_per_block](d_array, a, b, Nsize)
        d_array.copy_to_host(array)
        return array
    def midpoints(self):
        Nsize=self.N+1
        array=np.zeros(Nsize)
        mesh=self.mesh_points()
        thread, block = init_threads_blocks(128, Nsize)
        d_array=cuda.to_device(array)
        p_midpoints[block,thread](mesh,array,Nsize)
        d_array.copy_to_host(array)
        return array
    def silengths(self):
        Nsize=self.N+1
        array=np.zeros(Nsize)
        mesh=self.mesh_points()
        helper_interval_lengths(mesh,array,Nsize)
        return array
    def time_points(self):
        array=np.zeros(self.M+1)
        t_0,t_m,M=self.t_0,self.t_m,self.M+1
        helper_linspace(array,t_0,t_m,M)
        return array
    def delta_t(self):
        Nsize=self.M
        array=np.zeros(Nsize)
        time=self.time_points()
        d_time=cuda.to_device(time)
        helper_interval_lengths(d_time,array,Nsize)
        return array

        


@cuda.jit('void(float64[:],float64,float64,float64)')
def p_linspace(array, a, b, Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    step=(b-a)/(Nsize-1)
    if(idx<Nsize):
        array[idx]=a+(idx*step)


@cuda.jit('void(float64[:], float64[:], float64)')
def p_midpoints(mesh, mid, Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        mid[idx]=(mesh[idx]+mesh[idx+1])/2.0

@cuda.jit('void(float64[:], float64[:], float64)')
def p_interval_lengths(mesh, length, Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    if(idx<Nsize):
        length[idx]=(mesh[idx]-mesh[idx+1])

def helper_linspace(array, a, b, Nsize):
    threads_per_block, blocks_per_grid= init_threads_blocks(32, Nsize)
    d_array=cuda.to_device(array)
    p_linspace[blocks_per_grid,threads_per_block](d_array, a, b, Nsize)
    d_array.copy_to_host(array)

def helper_interval_lengths(array, return_array, Nsize):
    threads_per_block, blocks_per_grid= init_threads_blocks(32, Nsize)
    d_r_array=cuda.to_device(return_array)
    d_array=cuda.to_device(array)
    p_interval_lengths[blocks_per_grid,threads_per_block](d_array,return_array,Nsize)
    d_r_array.copy_to_host(return_array)

def init_threads_blocks(threads, Nsize):
    threads_per_block=threads
    blocks_per_grid=(Nsize+threads_per_block-1)//threads_per_block
    return threads_per_block, blocks_per_grid

m=Mesh(-4,4,1000,0,1,300)
print(m.mesh_points(),
m.time_points(),
m.midpoints(),
m.silengths(),
m.delta_t())