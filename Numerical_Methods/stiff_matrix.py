from xml.etree.ElementTree import XML
import numpy as np
import scipy
from scipy.linalg import toeplitz
from scipy.sparse import diags
from math import gamma
from mesh import Mesh



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
        col=[(1-self.gamma)*(1.5**self.beta)-(2-self.gamma)*(.5)**self.beta, (1-self.gamma)*((.5**self.beta)+(2.5**self.beta)-2*(1.5)**self.beta), ((1-self.gamma)*((-i-.5)**self.beta+(-i+1.5)**self.beta-2*(-i+.5)**self.beta) for i in range(-2,-self.N-2,-1))]
        row=[(1-self.gamma)*(1.5**self.beta)-(2-self.gamma)*(.5)**self.beta, (1+self.gamma)*(.5**self.beta)-(self.gamma)*(1.5**self.beta), (self.gamma*(2*(i-.5)**self.beta-(i+.5)**self.beta-(i-1.5**self.beta)) for i in range(2,self.N))]
        BL=toeplitz(c=col, r=row)
        return BL
    def BR(self):
        BR=np.zeros((self.N,self.N))
        col=[self.gamma(1.5)**self.beta-(1+self.gamma)*(.5)**self.beta, (2-self.gamma)*(.5)**self.beta-(1-self.gamma)*(1.5)**self.beta, ((1-self.gamma)*(2*(-i-.5)**self.beta-(-i-1.5)**self.beta-(-i+.5)**self.beta) for i in range(-2,-self.N-2,-1))]
        row=[self.gamma(1.5)**self.beta-(1+self.gamma)*(.5)**self.beta, self.gamma*((2.5)**self.beta-2*(1.5)**self.beta+(.5)**self.beta), ((self.gamma)*((i+1.5)**self.beta-2*(i+.5)**self.beta+(i-.5)**self.beta) for i in range(2,self.N))]
        BR=toeplitz(c=col, r=row)
        return BR
    def B(self,t):
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m=np.zeros(self.N+1)
        K_m=coeff(x=self.mid[0:self.N+1],t=t)
        B=np.ones(self.N,self.N)
        B[:]=(1/(gamma(self.beta)*self.h[0]))*(np.matmul(diags(K_m[1:],k=0).toarray(),self.BL()[:])+np.matmul(diags(K_m[:self.N+1]).toarray(),self.BR()[:]))
        return B




