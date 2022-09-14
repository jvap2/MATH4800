import cupy as cp
from d_mesh_d import Mesh
from math import gamma


class B_mat():
    def __init__(self,a,b,N,t_0,t_m,M,alpha, gamma):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.alpha=alpha
        self.gamma=gamma
        self.N=N
    def c_plus(self,x=0,t=0):
        cplus=self.gamma*.01*cp.ones(self.N+1)
        return cplus
    def c_minus(self,x=0,t=0):
        cmin=(1-self.gamma)*.01*cp.ones(self.N+1)
        return cmin
    def g(self):
        g=cp.zeros(shape=self.N+1, dtype=cp.float64)
        k=cp.linspace(0,self.N,self.N+1)
        for (i,point) in enumerate(k):
            try:
                g[i]=gamma(point-self.alpha)/(gamma(-self.alpha)*gamma(point+1))
            except OverflowError:
                g[i]=0
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
        diag=cp.zeros(self.N+1)
        lower_diag=cp.zeros(self.N)
        upper_diag=cp.zeros(self.N)
        diag[0]=1
        diag[self.N]=1
        diag[1:self.N]=1+(self.eps()[1:self.N]+self.eta()[1:self.N])*self.g()[1]
        lower_diag[:self.N-1]=self.eps()[2:self.N]*self.g()[2]+self.eta()[2:self.N]*self.g()[0]
        lower_diag[self.N-1]=0
        upper_diag[1:]=self.eps()[1:self.N-1]*self.g()[0]+self.eta()[1:self.N-1]*self.g()[2]
        upper_diag[0]=0
        B=cp.diag(diag,k=0)+cp.diag(lower_diag,k=-1)+cp.diag(upper_diag,k=1)
        for i in range(2,self.N):
            ud=cp.zeros(self.N-i+1)
            ld=cp.zeros(self.N-i+1)
            ud[0]=0
            ud[1:]=self.eta()[1:self.N-i]*self.g()[i+1]
            ld[0:self.N-i]=self.eps()[i+1:self.N]*self.g()[i+1]
            ld[self.N-i]=0
            B+=cp.diag(ud,k=i)
            B+=cp.diag(ld,k=-i)
        return B




        
