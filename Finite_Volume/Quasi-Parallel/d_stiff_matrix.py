from xml.etree.ElementTree import XML
import cupy as cp
from cupy import diag
import cupyx.scipy
from cupyx.scipy.linalg import toeplitz
from cupyx.scipy.sparse import diags
from math import gamma
from cupyx.scipy.sparse.linalg import LinearOperator
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
        beta=self.beta
        g=self.gamma
        h=self.h[0]
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_1=cp.ones(self.N+1)*.01
        K_m=cp.zeros(self.N+1)
        K_m=coeff(x=self.mid[0:self.N+1],t=t)
        k_left_non_lin=lambda x: 1+x
        K_m_ss=cp.ones(self.N+1)
        K_m_ss_nonlin=k_left_non_lin(self.mid)
        col_linspace=cp.linspace(3,self.N-1, self.N-3)
        j=cp.linspace(3,self.N, self.N-2)
        col_min=cp.empty(shape=self.N)
        col_plus=cp.empty(shape=self.N)
        col_min[3:]=((col_linspace+1.5)**beta)*(-2*(col_linspace**2)-6*col_linspace+((beta**2)/3)+beta-(23/6))+(2/3)*((col_linspace+.5)**beta)*(12*(col_linspace**2)+12*col_linspace-2*(beta**2)-6*beta-1)+\
            ((col_linspace-.5)**beta)*(-12*(col_linspace**2)+12*col_linspace+2*(beta**2)+6*beta+1)+(2/3)*((col_linspace-1.5)**beta)*(12*(col_linspace**2)-36*col_linspace-2*beta**2-6*beta+23)+\
            ((col_linspace-2.5)**beta)*(-2*(col_linspace**2)+10*col_linspace+((beta**2)/3)+beta-(71/6))
        col_plus[3:]=((col_linspace+2.5)**beta)*(2*(col_linspace**2)+10*col_linspace-(beta**2/3)-beta+(71/6))+(2/3)*((col_linspace+1.5)**beta)*(-12*(col_linspace**2)-36*col_linspace+2*beta**2+6*beta-23)+\
            ((col_linspace+.5)**beta)*(12*(col_linspace**2)+12*col_linspace-2*beta**2-6*beta-1)+(2/3)*((col_linspace-.5)**beta)*(-12*(col_linspace**2)+12*col_linspace+2*beta**2+6*beta+1)+\
            ((col_linspace-1.5)**beta)*(2*(col_linspace**2)-6*col_linspace-((beta**2)/3)-beta+(23/6))
        col_plus[2]=((4.5)**beta)*((239/6)-((beta**2)/3)-beta)+(2/3)*((3.5)**beta)*(-143+2*(beta**2)+6*beta)+\
            ((2.5)**beta)*(71-2*(beta**2)-6*beta)+(2/3)*((1.5)**beta)*(-23+2*(beta**2)+6*beta)+\
            ((.5)**beta)*(-(1/6)-((beta**2)/3)-beta)
        col_min[2]=((3.5**beta))*(((beta**2)/3)+beta-(143/6))+((2.5)**beta)*((-4*(beta**2)/3)-4*beta+(142/3))+((1.5)**beta)*(2*(beta**2)+6*beta-23)+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
        col_plus[1]=((3.5)**beta)*((-(beta**2)/3)-beta+(143/6))+((2.5)**beta)*((4*(beta**2)/3)+4*beta-(142/3))+((1.5)**beta)*(-2*(beta**2)-6*beta+23)+((.5)**beta)*((4*(beta**2)/3)+4*beta+(2/3))
        col_min[1]=((2.5)**beta)*(((beta**2)/3)+beta-(71/6))+((1.5)**beta)*((-4*(beta**2)/3)-4*beta+(46/3))+((.5)**beta)*(2*(beta**2)+6*beta+1)
        col_plus[0]=((2.5)**beta)*((-(beta**2)/3)-beta+(71/6))+((1.5)**beta)*((4*(beta**2)/3)+4*beta-(46/3))+((.5)**beta)*(-2*(beta**2)-6*beta-1)
        col_min[0]=((1.5)**beta)*(((beta**2)/3)+beta-(23/6))+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
        row_plus=cp.zeros(shape=self.N)
        row_min=cp.zeros(shape=self.N)
        row_min[0]=((1.5)**beta)*(((beta**2)/3)+beta-(23/6))+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
        row_plus[0]=((2.5)**beta)*((-(beta**2)/3)-beta+(71/6))+((1.5)**beta)*((4*(beta**2)/3)+4*beta-(46/3))+((.5)**beta)*(-2*(beta**2)-6*beta-1)
        row_min[1]=((.5)**beta)*(((beta**2)/3)+beta+(1/6))
        row_plus[1]=((.5)**beta)*((4*(beta**2)/3)+4*beta+(2/3))+((1.5)**beta)*((-(beta**2)/3)-beta+(23/6))
        row_plus[2]=((.5)**beta)*((-(beta**2)/3)-beta-(1/6))
        B_L_Plus=toeplitz(c=col_plus,r=row_plus)
        B_L_Min=toeplitz(c=col_min,r=row_min)
        B_L_Min[:,0],B_L_Plus[:,0],B_L_Min[:,1],B_L_Plus[:,0]=0,0,0,0
        B_L_Min[0,0]=(-(.5)**beta)*(6*beta**2+13*beta+3.5)
        B_L_Plus[0,0]=((1.5)**beta)*(6*beta**2+3*beta-4.5)
        B_L_Min[0,1]=((.5)**beta)*(3*beta**2+5*beta-.5)
        B_L_Plus[0,1]=((1.5)**beta)*(-3*beta**2+3*beta+4.5)
        B_L_Min[1,0]=((1.5)**beta)*(-6*beta**2-3*beta+4.5)
        B_L_Plus[1,0]=((.5)**beta)*((4/3)*beta**2+4*beta+(2/3))-((2.5)**beta)*(-6*beta**2+7*beta+.5)
        B_L_Min[1,1]=((1.5)**beta)*(3*beta**2-3*beta-4.5)
        B_L_Plus[1,1]=(-(.5)**beta)*(2*beta**2+6*beta+1)+((2.5)**beta)*(-3*beta**2+11*beta-3.5)
        B_L_Min[2,0]=(-(.5)**beta)*((4/3)*beta**2+4*beta+(2/3))+((2.5)**beta)*(-6*beta**2+7*beta+.5)
        B_L_Min[2,1]=-B_L_Plus[1,1]
        B_L_Plus[2,1]=((.5)**beta)*((4/3)*beta**2+4*beta+(2/3))-((1.5)**beta)*(2*beta**2+6*beta-23)+((3.5)**beta)*(-3*beta**2+19*beta-23.5)
        B_L_Min[3,1]=-B_L_Plus[2,1]
        B_L_Plus[2:,0]=((j-1.5)**beta)*((4/3)*(beta**2)+4*beta+24*j-8*(j**2)-(46/3))-((j+.5)**beta)*(6*(beta**2)-13*beta+10*(beta*j)+14*j-6*(j**2)-3.5)+\
            -((j-2.5)**beta)*((1/3)*(beta**2)+beta-2*(j**2)+10*j-(71/6))
        B_L_Min[3:,0]=((j[1:]-3.5)**beta)*((1/3)*(beta**2)+beta-(2*j[1:]**2)+14*j[1:]-(143/6))-((j[1:]-2.5)**beta)*((4/3)*(beta**2)+4*beta+40*j[1:]-(8*j[1:]**2)-(143/3))+\
            ((j[1:]-.5)**beta)*(-6*(beta**2)-23*beta+10*(beta*j[1:])+26*j[1:]-6*(j[1:]**2)-23.5)
        B_L_Plus[3:,1]=(-(j[1:]-3.5)**beta)*((1/3)*beta**2+beta-2*j[1:]**2+14*j[1:]-(143/6))+((j[1:]-2.5)**beta)*((4/3)*beta**2+4*beta+40*j[1:]-8*j[1:]**2-(142/3))+\
            ((j[1:]+.5)**beta)*(-3*beta**2-5*beta+8*beta*j[1:]+10*j[1:]-6*j[1:]**2+.5)-((j[1:]-1.5)**beta)*(2*beta**2+6*beta-12*j[1:]**2+36*j[1:]-23)
        B_L_Min[4:,1]=((j[2:]-4.5)**beta)*((1/3)*beta**2+beta-2*j[2:]**2+18*j[2:]-(239/6))-((j[2:]-3.5)**beta)*((4/3)*beta**2+4*beta+56*j[2:]-8*j[2:]**2-(286/3))-\
            ((j[2:]-.5)**beta)*(-3*beta**2-13*beta+8*beta*j[2:]+22*j[2:]-6*j[2:]**2-15.5)+((j[2:]-2.5)**beta)*(2*beta**2+6*beta-12*j[2:]**2+60*j[2:]-71)
        B_L_Plus[self.N-1,self.N-1]=((1.5)**beta)*((4/3)*beta**2+4*beta-(46/3))-((2.5)**beta)*((1/3)*beta**2+beta-(71/6))
        B_L_Min[self.N-1,self.N-1]=((1.5)**beta)*((1/3)*beta**2+beta-(23/6))-((.5)**beta)*((4/3)*beta**2+4*beta+(2/3))
        B_L_Plus[self.N-2,self.N-1]=-B_L_Min[self.N-1,self.N-1]
        B_L_Min[self.N-2,self.N-1]=((.5)*beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Plus[self.N-3,self.N-1]=-B_L_Min[self.N-2,self.N-1]
        B_L_Plus[self.N-1,self.N-2]=(-(1.5)**beta)*(2*beta**2+6*beta-23)+((2.5)**beta)*((4/3)*(beta**2)+4*beta-(142/3))-((3.5)**beta)*((1/3)*beta**2+beta-(143/6))
        B_L_Min[self.N-1,self.N-2]=((.5)**beta)*(2*beta**2+6*beta+1)-((1.5)**beta)*((4/3)*beta**2+4*beta-(46/3))-((2.5)**beta)*((1/3)*beta**2+beta-(71/6))
        B_L_Plus[self.N-2,self.N-2]=-B_L_Min[self.N-1,self.N-2]
        B_L_Min[self.N-2,self.N-2]=((1.5)**beta)*((1/3)*(beta**2)+beta-(23/6))-((.5)**beta)*((4/3)*(beta)**2+4*beta+(2/3))
        B_L_Plus[self.N-3,self.N-2]=-B_L_Min[self.N-2,self.N-2]
        B_L_Min[self.N-3,self.N-2]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Plus[self.N-4,self.N-2]=-B_L_Min[self.N-3,self.N-2]
        constant=(self.gamma*(h**(beta-1)))/(2*gamma(beta+3))
        K_plus_diag=constant*diag(K_m_ss[1:],k=0)
        K_min_diag=constant*diag(K_m_ss[:-1],k=0)
        B=cp.matmul(K_plus_diag,B_L_Plus)+cp.matmul(K_min_diag,B_L_Min)
        return B
    def Linear_Left_Deriv(self):
        n=cp.linspace(2,self.N-1, self.N-2)
        k_left_non_lin=lambda x: 1+x
        K_m_ss=cp.ones(self.N+1)
        K_m_ss_nonlin=k_left_non_lin(self.mesh.midpoints())
        col_plus,row_plus=cp.zeros(shape=(self.N)),cp.zeros(shape=(self.N))
        col_min,row_min=cp.zeros(shape=(self.N)),cp.zeros(shape=(self.N))
        col_plus[0]=(-2*(.5**self.beta)+(1.5)**self.beta)
        row_plus[0]=(-2*(.5**self.beta)+(1.5)**self.beta)
        row_plus[1]=(.5)**self.beta
        col_min[0]=-(.5)**self.beta
        row_min[0]=-(.5)**self.beta
        col_plus[1]=(-2*(1.5**self.beta)+(2.5**self.beta)+(.5**self.beta))
        col_min[1]=-(-2*(.5**self.beta)+(1.5**self.beta))
        col_plus[2:]=(-2*((n+.5)**self.beta)+((n+1.5)**self.beta)+((n-.5)**self.beta))
        col_min[2:]=(2*((n-.5)**self.beta)-((n+.5)**self.beta)-((n-1.5)**self.beta))
        const=self.gamma/(gamma(self.beta+1)*self.h[0]**(1-self.beta))
        K_plus_diag=const*diag(K_m_ss[1:],k=0)
        K_min_diag=const*diag(K_m_ss[:self.N],k=0)
        B_L_Plus=toeplitz(c=col_plus,r=row_plus)
        B_L_Min=toeplitz(c=col_min,r=row_min)
        B=cp.matmul(K_plus_diag,B_L_Plus)+cp.matmul(K_min_diag,B_L_Min)
        return B




b=StiffMatrix(0,1,10,0,1,5,1,.7)
print(b.Cubic_Left_Deriv()[:,0:2])