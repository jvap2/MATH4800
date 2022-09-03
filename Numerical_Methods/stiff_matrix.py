from xml.etree.ElementTree import XML
import numpy as np
from math import gamma
from mesh import Mesh
import multiprocessing


class StiffMatrix(Mesh):
    def __init__(self,a,b,N, gamma, beta):
        super().__init__(a,b,N)
        self.gamma=gamma
        self.beta=beta
        self.x=self.mesh_points()
        self.mid=self.midpoints()
        self.h=self.silengths()
    def construct_coeff(self,t=0):
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_pos=np.zeros(self.N)
        K_m_neg=np.zeros(self.N)
        K_m_pos[:]=coeff(x=self.mid[0:self.N],t=t)
        K_m_neg[:]=coeff(x=self.mid[1:self.N+1],t=t)
        return K_m_pos,K_m_neg
    def BL(self):
        BL=np.zeros((self.N,self.N))
        neg_list=[(1-self.gamma)*((-i-.5)**self.beta+(-i+1.5)**self.beta-2*(-i+.5)**self.beta) for i in range(-2,-self.N-2,-1)]
        pos_list=[self.gamma*(2*(i-.5)**self.beta-(i+.5)**self.beta-(i-1.5**self.beta)) for i in range(2,self.N)]
        i=0
        for i,(neg_item, pos_item) in enumerate(zip(neg_list,pos_list)):
            BL+=np.diag(np.repeat(neg_item,repeats=self.N-2-i),k=-2-i)+np.diag(np.repeat(pos_item, repeats=self.N-2-i),k=i+2)
        lower_diag=(1-self.gamma)*((.5**self.beta)+(2.5**self.beta)-2*(1.5)**self.beta)
        upper_diag=(1+self.gamma)*(.5**self.beta)-(self.gamma)*(1.5**self.beta)
        diag=(1-self.gamma)*(1.5**self.beta)-(2-self.gamma)*(.5)**self.beta
        BL=np.diag(np.repeat(lower_diag, repeats=self.N-1),k=-1)+np.diag(np.repeat(diag,repeats=self.N),k=0)+np.diag(np.repeat(upper_diag,repeats=self.N-1),k=1)
        return BL
    def BR(self):
        BR=np.zeros((self.N,self.N))
        neg_list=[(1-self.gamma)*(2*(-i-.5)**self.beta-(-i-1.5)**self.beta-(-i+.5)**self.beta) for i in range(-2,-self.N-2,-1)]
        pos_list=[(self.gamma)*((i+1.5)**self.beta-2*(i+.5)**self.beta+(i-.5)**self.beta) for i in range(2,self.N)]
        lower_diag=(2-self.gamma)*(.5)**self.beta-(1-self.gamma)*(1.5)**self.beta
        diag=self.gamma(1.5)**self.beta-(1+self.gamma)*(.5)**self.beta
        upper_diag=self.gamma*((2.5)**self.beta-2*(1.5)**self.beta+(.5)**self.beta)
        for i,(neg_item, pos_item) in enumerate(zip(neg_list,pos_list)):
            BR+=np.diag(np.repeat(neg_item, repeats=self.N-2-i),k=-2-i)+np.diag(np.repeat(pos_item, repeats=self.N-2-i), k=i+2)
        BR=np.diag(np.repeat(lower_diag, repeats=self.N-1),k=-1)+np.diag(np.repeat(diag, repeats=self.N),k=0)+np.diag(np.repeat(upper_diag, repeats=self.N-1),k=1)
        return BR
    def B(self):
        B=np.ones(self.N,self.N)
        B=(1/(gamma(self.beta)*self.h[0]))*(np.matmul(np.diag(self.construct_coeff()[0],k=0),self.BL)+np.matmul(np.diag(self.construct_coeff()[1]),self.BR))
        return B




