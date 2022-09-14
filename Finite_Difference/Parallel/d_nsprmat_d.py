import cupy as cp
from d_mesh_d import Mesh
from math import gamma


class A_mat():
    def __init__(self,a,b,N,t_0,t_m,M,alpha):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.alpha=alpha
    def c_plus(self,x=0,t=0):
        cplus=.01*cp.ones(self.N+1)
        return cplus
    def c_minus(self,x=0,t=0):
        cmin=.01*cp.ones(self.N+1)
        return cmin
    def g(self):
        g=cp.zeros(self.N+1)
        k=cp.linspace(0,self.N,self.N+1)
        g[:]=gamma(k[:]-self.alpha)/(gamma(-self.alpha)*gamma(k[:]+1))
        return g
    def eps(self):
        epsilon=cp.zeros(self.N-1)
        epsilon[:]=self.c_plus()[1:self.N]*self.mesh.delta_t()/(self.mesh.silengths()[0]**self.alpha)
        return epsilon
    def eta(self):
        eta=cp.zeros(self.N-1)
        eta[:]=self.c_minus()[1:self.N]*self.mesh.delta_t()/(self.mesh.silengths()[0]**self.alpha)
        return eta
    def Construct(self,x=0,t=0):
        A=cp.zeros(shape=(self.N+1,self.N+1))
        A[0,0],A[self.N,self.N]=1,1

        
