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
        array=np.zeros(self.N+2)
        b,a,Nsize=self.b,self.a,self.N+2
        helper_linspace(array,a,b,Nsize)
        print(array)
        return array
    def time_points(self):
        array=np.zeros(self.M+1)
        t_0,t_m,M=self.t_0,self.t_m,self.M+1
        helper_linspace(array,t_0,t_m,M)
        print(array)
        return array

        


@cuda.jit('void(float64[:],float64,float64,float64)')
def p_linspace(array, a, b, Nsize):
    idx=cuda.threadIdx.x+(cuda.blockDim.x*cuda.blockIdx.x)
    step=(b-a)/(Nsize-1)
    if(idx<Nsize):
        array[idx]=a+(idx*step)
def helper_linspace(array, a, b, Nsize):
    t=128
    b=(Nsize+t-1)//t
    d_array=cuda.to_device(array)
    p_linspace[b,t](d_array, a, b, Nsize)
    d_array.copy_to_host(array)
    

m=Mesh(-4,4,8,0,1,3)
m.mesh_points()
m.time_points()