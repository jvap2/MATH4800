import cupy as cp
from d_mesh_d import Mesh
from math import gamma


class B_mat():
    def __init__(self,a,b,N,t_0,t_m,M,alpha, gamma):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.alpha=alpha
        self.gamma=gamma
    def c_plus(self,x=0,t=0):
        cplus=self.gamma*.01*cp.ones(self.N+1)
        return cplus
    def c_minus(self,x=0,t=0):
        cmin=(1-self.gamma)*.01*cp.ones(self.N+1)
        return cmin
    def g(self):
        g=cp.zeros(self.N+1)
        k=cp.linspace(0,self.N,self.N+1)
        g[:]=gamma(k[:]-self.alpha)/(gamma(-self.alpha)*gamma(k[:]+1))
        return g
    def eps(self):
        epsilon=cp.zeros(self.N+1)
        epsilon[:]=self.c_plus()[:]*self.mesh.delta_t()/(self.mesh.silengths()[0]**self.alpha)
        return epsilon
    def eta(self):
        eta=cp.zeros(self.N+1)
        eta[:]=self.c_minus()[:]*self.mesh.delta_t()/(self.mesh.silengths()[0]**self.alpha)
        return eta
    def Construct(self,x=0,t=0):
        B=cp.zeros(shape=(self.N+1,self.N+1))
        diag=cp.diag(cp.ndarray([1,1+(self.eps()[1:self.N]+self.eta()[1:self.N])*self.g()[1],1]),k=0)
        lower_diag=cp.diag(cp.ndarray([self.eps()[2:self.N]*self.g()[2]+self.eta()[2:self.N]*self.g()[0],0]),k=-1)
        upper_diag=cp.diag(cp.ndarray([0,self.eps()[1:self.N-1]*self.g()[0]+self.eta()[1:self.N-1]*self.g()[2]]),k=1)
        B=diag+lower_diag+upper_diag
        for i in range(2,self.N):
            ud=cp.diag(cp.ndarray([0,self.eta()[1:self.N-i]*self.g()[i+1]]),k=i)
            ld=cp.diag(cp.ndarray([self.eps()[i+1:self.N]*self.g()[i+1],0]),k=-i)
            B+=(ud+ld)
        return B




        
