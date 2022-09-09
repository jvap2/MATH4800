from xml.etree.ElementTree import XML
import cupy as cp
import cupyx.scipy
from cupyx.scipy.linalg import toeplitz
from cupyx.scipy.sparse import diags
from math import gamma
from d_mesh import Mesh



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
        col_linspace=cp.linspace(-2,1-self.N,self.N-2)
        row_linspace=cp.linspace(2,self.N-1, self.N-2)
        col,row=cp.zeros(shape=(self.N)),cp.zeros(shape=(self.N))
        col[0],row[0]=(1-self.gamma)*(3/2)**self.beta-(2-self.gamma)*(1/2)**self.beta,(1-self.gamma)*(3/2)**self.beta-(2-self.gamma)*(1/2)**self.beta
        col[1],row[1]=(1-self.gamma)*((1/2)**self.beta+(5/2)**self.beta-2*(3/2)**self.beta),(1+self.gamma)*(1/2)**self.beta-(self.gamma)*(3/2)**self.beta
        col[2:]=(1-self.gamma)*((-col_linspace-(1/2))**self.beta+(-col_linspace+(3/2))**self.beta-2*(-col_linspace+(1/2))**self.beta)
        row[2:]=self.gamma*(2*(row_linspace-(1/2))**self.beta-(row_linspace+(1/2))**self.beta-(row_linspace-(3/2))**self.beta)
        BL=toeplitz(c=col,r=row)
        return BL
    def BR(self):
        col_linspace=cp.linspace(-2,1-self.N,self.N-2)
        row_linspace=cp.linspace(2,self.N-1, self.N-2)
        col,row=cp.zeros(shape=(self.N)),cp.zeros(shape=(self.N))
        col[0],row[0]=self.gamma*(3/2)**self.beta-(1+self.gamma)*(1/2)**self.beta,self.gamma*(3/2)**self.beta-(1+self.gamma)*(1/2)**self.beta
        col[1],row[1]=(2-self.gamma)*(1/2)**self.beta-(1-self.gamma)*(3/2)**self.beta,self.gamma*((5/2)**self.beta-2*(3/2)**self.beta+(1/2)**self.beta)
        col[2:]=(1-self.gamma)*(2*(-col_linspace-(1/2))**self.beta-(-col_linspace-(3/2))**self.beta-(-col_linspace+(1/2))**self.beta)
        row[2:]=self.gamma*((row_linspace+(3/2))**self.beta-2*(row_linspace+(1/2))**self.beta+(row_linspace-(1/2))**self.beta)
        BR=toeplitz(c=col, r=row)
        return BR
    def B(self,t):
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_1=.01*cp.ones(self.N+1)
        K_m=cp.zeros(self.N+1)
        K_m=coeff(x=self.mid[0:self.N+1],t=t)
        B=cp.ones((self.N,self.N))
        B[:]=(1/(gamma(self.beta+1)*(self.h[0])**(1-self.beta)))*(cp.matmul(cp.diag(K_m_1[:self.N],k=0),self.BL())+cp.matmul(cp.diag(K_m_1[1:],k=0),self.BR()))
        return B




