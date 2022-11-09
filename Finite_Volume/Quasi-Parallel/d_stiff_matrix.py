from xml.etree.ElementTree import XML
import cupy as cp
from cupy import diag
import cupyx.scipy
from cupyx.scipy.linalg import toeplitz
from cupyx.scipy.sparse import diags
from math import gamma
from cupyx.scipy.sparse.linalg import LinearOperator

from pyparsing import col
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
    def B_1(self):
        K_m_1=.01*cp.ones(self.N+1)
        B=cp.ones((self.N,self.N))
        B[:]=(1/(gamma(self.beta+1)*(self.h[0])**(1-self.beta)))*(cp.matmul(cp.diag(K_m_1[:self.N],k=0),self.BL())+cp.matmul(cp.diag(K_m_1[1:],k=0),self.BR()))
        return B
    def B(self,t):
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m=cp.zeros(self.N+1)
        K_m=coeff(x=self.mid[0:self.N+1],t=t)
        B=cp.ones((self.N,self.N))
        B[:]=(1/(gamma(self.beta+1)*(self.h[0])**(1-self.beta)))*(cp.matmul(cp.diag(K_m[:self.N],k=0),self.BL())+cp.matmul(cp.diag(K_m[1:],k=0),self.BR()))
        return B
    def Cubic_Left_Deriv(self,t=0):
        h=self.h[0]
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_1=cp.ones(self.N+1)*.01
        K_m=cp.zeros(self.N+1)
        K_m=coeff(x=self.mid[0:self.N+1],t=t)
        K_m_ss=cp.ones(self.N)
        col_linspace=cp.linspace(3,self.N-1, self.N-3)
        col_min=cp.empty(shape=self.N)
        col_plus=cp.empty(shape=self.N)
        col_min[3:]=((col_linspace+1.5)**self.beta)*(-2*(col_linspace**2)-6*col_linspace+((self.beta**2)/3)+self.beta-(23/6))+(2/3)*((col_linspace+.5)**self.beta)*(12*(col_linspace**2)+12*col_linspace-2*self.beta**2-6*self.beta-1)+\
            ((col_linspace-.5)**self.beta)*(-12*(col_linspace**2)+12*col_linspace+2*self.beta**2+6*self.beta+1)+(2/3)*((col_linspace-1.5)**self.beta)*(12*(col_linspace**2)-36*col_linspace-2*self.beta**2-6*self.beta+23)+\
            ((col_linspace-2.5)**self.beta)*(-2*(col_linspace**2)+10*col_linspace+(self.beta**2/3)+self.beta-(71/6))
        col_plus[3:]=((col_linspace+2.5)**self.beta)*(2*(col_linspace**2)+10*col_linspace-(self.beta**2/3)-self.beta+(71/6))+(2/3)*((col_linspace+1.5)**self.beta)*(-12*(col_linspace**2)-36*col_linspace+2*self.beta**2+6*self.beta-23)+\
            ((col_linspace+.5)**self.beta)*(12*(col_linspace**2)+12*col_linspace-2*self.beta**2-6*self.beta-1)+(2/3)*((col_linspace-.5)**self.beta)*(-12*(col_linspace**2)+12*col_linspace+2*self.beta**2+6*self.beta+1)+\
            ((col_linspace-1.5)**self.beta)*(2*(col_linspace**2)-6*col_linspace-(self.beta**2/3)-self.beta+(23/6))
        col_plus[2]=((4.5)**self.beta)*((239/6)-((self.beta**2)/3)-self.beta)+(2/3)*((3.5)**self.beta)*(-143+2*(self.beta**2)+6*self.beta)+\
            ((2.5)**self.beta)*(71-2*(self.beta**2)-6*self.beta)+(2/3)*((1.5)**self.beta)*(-23+2*(self.beta**2)+6*self.beta)+\
            ((.5)**self.beta)*(-(1/6)-((self.beta**2)/3)-self.beta)
        col_min[2]=((3.5**self.beta))*(((self.beta**2)/3)+self.beta-(143/6))+((2.5)**self.beta)*((-4*(self.beta**2)/3)-4*self.beta+(142/3))+((1.5)**self.beta)*(2*(self.beta**2)+6*self.beta-23)+((.5)**self.beta)*((-4*(self.beta**2)/3)-4*self.beta-(2/3))
        col_plus[1]=((3.5)**self.beta)*((-(self.beta**2)/3)-self.beta+(143/6))+((2.5)**self.beta)*((4*(self.beta**2)/3)+4*self.beta-(142/3))+((1.5)**self.beta)*(-2*(self.beta**2)-6*self.beta+23)+((.5)**self.beta)*((4*(self.beta**2)/3)+4*self.beta+(2/3))
        col_min[1]=((2.5)**self.beta)*((self.beta**2/3)+self.beta-(71/6))+((1.5)**self.beta)*((-4*(self.beta**2)/3)-4*self.beta+(46/3))+((.5)**self.beta)*(2*(self.beta**2)+6*self.beta+1)
        col_plus[0]=((2.5)**self.beta)*((-(self.beta**2)/3)-self.beta+(71/6))+((1.5)**self.beta)*((4*(self.beta**2)/3)+4*self.beta-(46/3))+((.5)**self.beta)*(-2(self.beta**2)-6*self.beta-1)
        col_min[0]=((1.5)**self.beta)*(((self.beta**2)/3)+self.beta-(23/6))+((.5)**self.beta)*((-4(self.beta**2)/3)-4*self.beta-(2/3))
        row_plus=cp.zeros(shape=self.N)
        row_min=cp.zeros(shape=self.N)
        row_min[0]=((1.5)**self.beta)*(((self.beta**2)/3)+self.beta-(23/6))+((.5)**self.beta)*((-4(self.beta**2)/3)-4*self.beta-(2/3))
        row_plus[0]=((2.5)**self.beta)*((-(self.beta**2)/3)-self.beta+(71/6))+((1.5)**self.beta)*((4*(self.beta**2)/3)+4*self.beta-(46/3))+((.5)**self.beta)*(-2(self.beta**2)-6*self.beta-1)
        row_min[1]=((.5)**self.beta)*(((self.beta**2)/3)+self.beta+(1/6))
        row_plus[1]=((.5)**self.beta)*((4*(self.beta**2)/3)+4*self.beta+(2/3))+((1.5)**self.beta)*((-(self.beta**2)/3)-self.beta+(23/6))
        row_plus[2]=((.5)**self.beta)*((-(self.beta**2)/3)-self.beta-(1/6))
        B_L_Plus=toeplitz(c=col_plus,r=row_plus)
        B_L_Min=toeplitz(c=col_min,r=row_min)
        constant=(self.gamma*(h**(self.beta-1)))/(2*gamma(self.beta+3))
        K_plus_diag=constant*diag(K_m_ss,k=0)
        K_min_diag=constant*diag(K_m_ss,k=0)
        B=cp.matmul(K_plus_diag,B_L_Plus)+cp.matmul(K_min_diag,B_L_Min)
        return B




