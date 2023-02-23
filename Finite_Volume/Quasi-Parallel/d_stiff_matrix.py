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
    def __init__(self,a,b,N,gamma,beta,t_0=0,t_m=0,M=0):
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
    def B(self,t=0):
        coeff=lambda x: 1+x
        K_m=cp.zeros(self.N+1)
        K_m=coeff(x=self.mid[0:self.N+1])
        B=cp.ones((self.N,self.N))
        B[:]=(1/(gamma(self.beta+1)*(self.h[0])**(1-self.beta)))*(cp.matmul(cp.diag(K_m[:self.N],k=0),self.BL())+cp.matmul(cp.diag(K_m[1:],k=0),self.BR()))
        return B
    def Cubic_Left_Deriv(self,t=0):
        beta=self.beta
        g=self.gamma
        h=self.h[1]
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_1=cp.ones(self.N+1)*.01
        K_m=cp.zeros(self.N+1)
        K_m=coeff(x=self.mid[0:self.N+1],t=t)
        #k_left_non_lin=lambda x: 1+x**3
        k_left_non_lin=lambda x: (1+x)**2
        K_m_ss_nonlin=k_left_non_lin(self.mid[0:])
        col_linspace=cp.linspace(3,self.N-1, self.N-3)
        j=cp.linspace(3,self.N, self.N-2)
        j_1=cp.linspace(4,self.N, self.N-3)
        j_2=cp.linspace(5,self.N, self.N-4)
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
        B_L_Min[:,0],B_L_Plus[:,0],B_L_Min[:,1],B_L_Plus[:,1]=0,0,0,0
        B_L_Min[0,0]=(-1)*((.5)**beta)*(6*(beta**2)+13*beta+3.5)
        B_L_Plus[0,0]=((1.5)**beta)*(6*(beta**2)+3*beta-4.5)
        B_L_Min[0,1]=((.5)**beta)*((3*beta**2)+5*beta-.5)
        B_L_Plus[0,1]=((1.5)**beta)*(-3*(beta**2)+3*beta+4.5)
        B_L_Min[1,0]=((1.5)**beta)*(-6*(beta**2)-3*beta+4.5)
        B_L_Plus[1,0]=((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((2.5)**beta)*(-6*(beta**2)+7*beta+.5)
        B_L_Min[1,1]=-B_L_Plus[0,1]
        B_L_Plus[1,1]=((.5)**beta)*(-2*(beta**2)-6*beta-1)-((2.5)**beta)*(3*(beta**2)-11*beta+3.5)
        B_L_Min[2,0]=-B_L_Plus[1,0]
        B_L_Min[2,1]=-B_L_Plus[1,1]
        B_L_Plus[2,1]=((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((3.5)**beta)*(-3*beta**2+19*beta-23.5)
        B_L_Min[3,1]=-B_L_Plus[2,1]
        B_L_Plus[0,2]=(-1)*((1.5)**beta)*(-2*(beta**2)/3+beta+(1/6))
        B_L_Min[0,2]=((.5)**beta)*(-2*(beta**2)/3-beta+(1/6))
        B_L_Min[1,2]=-B_L_Plus[0,2]
        B_L_Plus[1,2]=((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))-((2.5)**beta)*(-2*(beta**2)/3+3*beta-23/6)
        B_L_Min[2,2]=-B_L_Plus[1,2]
        B_L_Plus[2,2]=((1.5)**beta)*(4*(beta**2)/3+4*beta-(46/3))-((.5**beta)*(2*beta**2+6*beta+1)+(3.5**beta)*(-2*(beta**2)/3+5*beta-(71/6)))
        B_L_Min[3,2]=-B_L_Plus[2,2]
        B_L_Plus[3,2]=((2.5)**beta)*(4*(beta**2)/3+4*beta-(142/3))+((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))\
            -((1.5**beta)*(2*(beta**2)+6*beta-23)+(4.5**beta)*(-2*(beta**2)/3+7*beta-(143/6)))
        B_L_Min[4,2]=-B_L_Plus[3,2]
        B_L_Plus[4:,2]=((j_2-3.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+56*j_2-(286/3))+((j_2-1.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+24*j_2-(46/3))-(((j_2-4.5)**beta)*((beta**2)/3+beta-2*j_2**2+18*j_2-(239/6))+((j_2-2.5)**beta)*(2*beta**2+6*beta-12*j_2**2+60*j_2-71)+((j_2+.5)**beta)*(-2*(beta**2)/3-beta+2*beta*j_2+2*j_2-2*j_2**2+(1/6)))
        B_L_Min[5:,2]=-B_L_Plus[4:self.N-1,2]
        B_L_Plus[2:,0]=((j-1.5)**beta)*((4/3)*(beta**2)+4*beta-8*j**2+24*j-(46/3))-((j-2.5)**beta)*((1/3)*(beta**2)+beta-2*j**2+10*j-(71/6))-\
            ((j+.5)**beta)*(-6*(beta**2)-13*beta+10*beta*j-6*j**2+14*j-3.5)
        B_L_Min[3:,0]=-B_L_Plus[2:self.N-1,0]
        B_L_Plus[3:,1]=-((j_1-3.5)**beta)*((1/3)*(beta**2)+beta-2*(j_1**2)+14*j_1-(143/6))+((j_1-2.5)**beta)*((4/3)*beta**2+4*beta+40*j_1-8*(j_1**2)-(142/3))+\
            ((j_1+.5)**beta)*(-3*beta**2-5*beta+8*beta*(j_1)+10*j_1-6*(j_1**2)+.5)-((j_1-1.5)**beta)*(2*beta**2+6*beta-12*j_1**2+36*j_1-23)
        B_L_Min[4:,1]=-B_L_Plus[3:self.N-1,1]
        B_L_Plus[self.N-1,self.N-1]=((1.5)**beta)*((4/3)*beta**2+4*beta-(46/3))-((2.5)**beta)*((1/3)*beta**2+beta-(71/6))
        B_L_Min[self.N-1,self.N-1]=((1.5)**beta)*((1/3)*beta**2+beta-(23/6))-((.5)**beta)*((4/3)*beta**2+4*beta+(2/3))
        B_L_Plus[self.N-2,self.N-1]=-B_L_Min[self.N-1,self.N-1]
        B_L_Min[self.N-2,self.N-1]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Plus[self.N-3,self.N-1]=-B_L_Min[self.N-2,self.N-1]
        B_L_Plus[self.N-1,self.N-2]=-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((2.5)**beta)*((4/3)*(beta**2)+4*beta-(142/3))-\
            ((3.5)**beta)*((1/3)*(beta**2)+beta-(143/6))
        B_L_Min[self.N-1,self.N-2]=((.5)**beta)*(2*(beta**2)+6*beta+1)-((1.5)**beta)*((4/3)*(beta**2)+4*beta-(46/3))+\
            ((2.5)**beta)*((1/3)*(beta**2)+beta-(71/6))
        B_L_Plus[self.N-2,self.N-2]=-B_L_Min[self.N-1,self.N-2]
        B_L_Min[self.N-2,self.N-2]=((1.5)**beta)*((1/3)*(beta**2)+beta-(23/6))-((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))
        B_L_Plus[self.N-3,self.N-2]=-B_L_Min[self.N-2,self.N-2]
        B_L_Min[self.N-3,self.N-2]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Plus[self.N-4,self.N-2]=-B_L_Min[self.N-3,self.N-2]
        B_L_Plus[self.N-1,self.N-3]=((3.5)**beta)*(4*(beta**2)/3+4*beta-(286/3))+((1.5)**beta)*(4*(beta**2)/3+4*beta-46/3)-((4.5**beta)*((beta**2)/3+beta-(239/6))+(2.5**beta)*(2*(beta**2)+6*beta-71))
        constant=(self.gamma*(h**(beta-1)))/(2*gamma(beta+3))
        K_plus_diag=constant*diag(K_m_ss_nonlin[1:],k=0)
        K_min_diag=constant*diag(K_m_ss_nonlin[:self.N],k=0)
        B=cp.matmul(K_plus_diag,B_L_Plus)+cp.matmul(K_min_diag,B_L_Min)
        return B

    def Cubic_Right_Deriv(self):
        mempool = cp.get_default_memory_pool()
        beta=self.beta
        g=self.gamma
        h=self.h[1]
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_1=cp.ones(self.N+1)*.01
        K_m=cp.zeros(self.N+1)
        # K_m=coeff(x=self.mid[0:self.N+1],t=t)
        k_left_non_lin=lambda x: 1+x
        K_m_ss=cp.ones(self.N+1)
        K_m_ss_nonlin=k_left_non_lin(self.mid)
        N=self.N
        j=cp.linspace(1,self.N-3, self.N-3)
        j_1=cp.linspace(1,self.N-2, self.N-2)
        q=cp.linspace(-2,-(self.N-1), self.N-2)
        col_min=cp.empty(shape=self.N)
        col_plus=cp.empty(shape=self.N)
        row_min=cp.empty(shape=self.N)
        row_plus=cp.empty(shape=self.N)
        row_plus[3:]=((-q[1:]+.5)**beta)*((-4*beta**2)/3-4*beta+8*q[1:]**2-8*q[1:]-(2/3))+((-q[1:]-1.5)**beta)*((-4*beta**2)/3-4*beta+8*q[1:]**2+24*q[1:]+(46/3))-\
            (((-q[1:]+1.5)**beta)*((-beta**2)/3-beta+2*q[1:]**2-6*q[1:]+(23/6))+((-q[1:]-.5)**beta)*(-2*beta**2-6*beta+12*q[1:]**2+12*q[1:]-1)+((-q[1:]-2.5)**beta)*((-beta**2)/3-beta+2*q[1:]**2+10*q[1:]+(71/6)))
        row_min[2:]=((-q+2.5)**beta)*((-beta**2)/3-beta+2*q**2-10*q+(71/6))+((-q+.5)**beta)*(-2*beta**2-6*beta+12*q**2-12*q-1)+((-q-1.5)**beta)*((-beta**2)/3-beta+2*q**2+6*q+(23/6))-\
            (((-q+1.5)**beta)*((-4*beta**2)/3-4*beta+8*q**2-24*q+(46/3))+((-q-.5)**beta)*((-4*beta**2)/3-4*beta+8*q**2+8*q-(2/3)))
        row_plus[2]=(2.5**beta)*((-4*beta**2)/3-4*beta+(142/3))+(.5**beta)*((-4*beta**2)/3-4*beta-(2/3))-((3.5**beta)*((-beta**2)/3-beta+(143/6))+(1.5**beta)*(-2*beta**2-6*beta+23))
        row_min[1]=-row_plus[2]
        row_plus[1]=(1.5**beta)*((-4*beta**2)/3-4*beta+(46/3))-((2.5**beta)*((-beta**2)/3-beta+71/6)+(.5**beta)*(-2*beta**2-6*beta-1))
        row_min[0],col_min[0]=-row_plus[1],-row_plus[1]
        row_plus[0]=(.5**beta)*((-4*beta**2)/3-4*beta-2/3)-(1.5**beta)*((-beta**2)/3-beta+23/6)
        col_plus[0]=row_plus[0]
        col_min[1]=-row_plus[0]
        col_plus[1]=(-1)*((.5)**beta)*((-beta**2)/3-beta-1/6)
        col_min[2]=-col_plus[1]
        B_R_Plus=toeplitz(c=col_plus,r=row_plus)
        B_R_Min=toeplitz(c=col_min,r=row_min)
        B_R_Plus[0:N-3,N-1]=(-(N-j+.5)**beta)*(6*(beta**2)+(13-10*N)*beta+10*beta*j+6*(j**2)+j*(14-12*N)+6*(N**2)-14*N+3.5)+\
            ((N-j-1.5)**beta)*((-4*beta**2)/3-4*beta+8*(j**2)+j*(24-16*N)+8*(N**2)-24*N+(46/3))-((N-j-2.5)**beta)*((-beta**2)/3-beta+2*(j**2)+j*(10-4*N)+2*(N**2)-10*N+71/6)
        B_R_Min[0:N-2,N-1]=((N-j_1-1.5)**beta)*((-beta**2)/3-beta+2*j_1**2+j_1*(6-4*N)+2*N**2-6*N+23/6)-((N-j_1-.5)**beta)*((-4*beta**2)/3-4*beta+8*j_1**2+(8-16*N)*j_1+8*N**2-8*N-2/3)+\
            ((N-j_1+1.5)**beta)*(6*beta**2+(3-10*N)*beta+10*beta*j_1+6*j_1**2+(2-12*N)*j_1+6*N**2-2*N-4.5)
        B_R_Plus[N-3,N-1]=(-(2.5)**beta)*(6*beta**2-7*beta-.5)+((.5)**beta)*((-4*beta**2)/3-4*beta-2/3)
        B_R_Min[N-2,N-1]=-B_R_Plus[N-3,N-1]
        B_R_Plus[N-2,N-1]=(-(1.5)**beta)*(6*beta**2+3*beta-4.5)
        B_R_Min[N-1,N-1]=-B_R_Plus[N-2,N-1]
        B_R_Plus[N-1,N-1]=(-(.5)**beta)*(6*beta**2+13*beta+3.5)
        B_R_Plus[0:N-4,N-2]=-((N-j[:-1]-3.5)**beta)*((-beta**2)/3-beta+2*j[:-1]**2+j[:-1]*(14-4*N)+2*N**2-14*N+143/6)+((N-j[:-1]-2.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-1]**2+40*j[:-1]-16*N*j[:-1]+8*N**2-40*N+142/3)-\
            ((N-j[:-1]-1.5)**beta)*(-2*beta**2-6*beta+12*j[:-1]**2+36*j[:-1]-24*j[:-1]*N+12*N**2-36*N+23)+((N-j[:-1]+.5)**beta)*(3*beta**2+beta*(8*j[:-1]-8*N+5)+6*j[:-1]**2+j[:-1]*(10-12*N)+6*N**2-10*N-.5)
        B_R_Min[0:N-3,N-2]=((N-j-2.5)**beta)*(-(beta**2)/3-beta+2*j**2+j*(10-4*N)+2*N**2-10*N+71/6)-((N-j-1.5)**beta)*((-4*beta**2)/3-4*beta+8*j**2+24*j-16*j*N+8*N**2-24*N+46/3)+\
            ((N-j-.5)**beta)*(-2*beta**2-6*beta+12*j**2+12*j-24*N*j+12*N**2-12*N-1)-((N-j+1.5)**beta)*(3*beta**2+beta*(8*j-8*N-3)+6*j**2-2*j*(6*N+1)+6*N**2+2*N-4.5)
        B_R_Min[N-1,N-2]=(-(1.5)**beta)*(3*beta**2-3*beta-4.5)
        B_R_Plus[N-1,N-2]=((.5)**beta)*(3*beta**2+5*beta-.5)
        B_R_Plus[N-2,N-2]=-B_R_Min[N-1,N-2]
        B_R_Min[N-2,N-2]=((.5)**beta)*(-2*beta**2-6*beta-1)-(2.5**beta)*(3*beta**2-11*beta+3.5)
        B_R_Plus[N-3,N-2]=-B_R_Min[N-2,N-2]
        B_R_Min[N-3,N-2]=(-(.5)**beta)*((-4*beta**2)/3-4*beta-2/3)+((1.5)**beta)*(-2*beta**2-6*beta+23)-(3.5**beta)*(3*beta**2-19*beta+23.5)
        B_R_Plus[N-4,N-2]=-B_R_Min[N-3,N-2]
        B_R_Plus[0:N-5,N-3]=((N-j[:-2]-3.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-2]**2-16*j[:-2]*N+56*j[:-2]+8*N**2-56*N+286/3)-((N-j[:-2]-4.5)**beta)*((-beta**2)/3-beta+2*j[:-2]**2+j[:-2]*(18-4*N)+2*N**2-18*N+239/6)-\
            ((N-j[:-2]-2.5)**beta)*(-2*beta**2-6*beta+12*j[:-2]**2+60*j[:-2]-24*j[:-2]*N+12*N**2-60*N+71)+((N-j[:-2]-1.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-2]**2+24*j[:-2]-16*N*j[:-2]+8*N**2-24*N+46/3)-\
            ((N-j[:-2]+.5)**beta)*((2*beta**2)/3+beta*(2*j[:-2]-2*N+1)+2*j[:-2]**2+j[:-2]*(2-4*N)+2*N**2-2*N-1/6)
        B_R_Min[0:N-4,N-3]=-((N-j[:-1]-2.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-1]**2+40*j[:-1]-16*j[:-1]*N+8*N**2-40*N+142/3)+((N-j[:-1]-3.5)**beta)*((-beta**2)/3-beta+2*j[:-1]**2+j[:-1]*(14-4*N)+2*N**2-14*N+143/6)+\
            ((N-j[:-1]-1.5)**beta)*(-2*beta**2-6*beta+12*j[:-1]**2+36*j[:-1]-24*j[:-1]*N+12*N**2-36*N+23)-((N-j[:-1]-.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-1]**2+8*j[:-1]-16*j[:-1]*N+8*N**2-8*N-2/3)+\
            ((N-j[:-1]+1.5)**beta)*((2*beta**2)/3+beta*(2*j[:-1]-2*N-1)+2*j[:-1]**2+j[:-1]*(-4*N-2)+2*N**2+2*N-1/6)
        B_R_Plus[N-1,N-3]=-((.5)**beta)*((2*beta**2)/3+beta-1/6)
        B_R_Min[N-1,N-3]=((1.5)**beta)*((2*beta**2)/3-beta-1/6)
        B_R_Plus[N-2,N-3]=-B_R_Min[N-1,N-3]
        B_R_Min[N-2,N-3]=-((.5)**beta)*((-4*beta**2)/3-4*beta-2/3)+((2.5)**beta)*((2*beta**2)/3-3*beta+23/6)
        B_R_Plus[N-3,N-3]=-B_R_Min[N-2,N-3]
        B_R_Min[N-3,N-3]=(.5**beta)*(-2*(beta**2)-6*beta-1)-(1.5**beta)*((-4*beta**2)/3-4*beta+46/3)+(3.5**beta)*((2*beta**2)/3-5*beta+71/6)
        B_R_Plus[N-4,N-3]=-B_R_Min[N-3,N-3]
        B_R_Min[N-4,N-3]=-((.5)**beta)*((-4*beta**2)/3-4*beta-(2/3))+(1.5**beta)*(-2*beta**2-6*beta+23)-(2.5**beta)*((-4*beta**2)/3-4*beta+142/3)+(4.5**beta)*((2*beta**2)/3-7*beta+143/6)
        B_R_Plus[N-5,N-3]=-B_R_Min[N-4,N-3]
        B_R_Min[0,2]=(4.5**beta)*((-beta**2)/3-beta+(239/6))+(2.5**beta)*(-2*beta**2-6*beta+71)-((3.5**beta)*((-4*beta**2)/3-4*beta+(286/3))+(1.5**beta)*((-4*beta**2)/3-4*beta+(46/3)))
        B_R_Min[0,1]=(3.5**beta)*((-beta**2)/3-beta+143/6)+(1.5**beta)*(-2*beta**2-6*beta+23)-(2.5**beta)*((-4*beta**2)/3-4*beta+142/3)
        B_R_Min[1,1]=(2.5**beta)*((-beta**2)/3-beta+(71/6))+(.5**beta)*(-2*beta**2-6*beta-1)-(1.5**beta)*((-4*beta**2)/3-4*beta+46/3)
        B_R_Plus[0,1]=-B_R_Min[1,1]
        B_R_Min[2,1]=(1.5**beta)*((-beta**2)/3-beta+(23/6))-(.5**beta)*((-4*beta**2)/3-4*beta-(2/3))
        B_R_Plus[1,1]=-B_R_Min[2,1]
        B_R_Min[3,1]=(.5**beta)*((-beta**2)/3-beta-(1/6))
        B_R_Plus[2,1]=-B_R_Min[3,1]
        B_R_Min[0,0]=(2.5**beta)*((-beta**2)/3-beta+(71/6))-(1.5**beta)*((-4*beta**2)/3-4*beta+(46/3))
        B_R_Min[1,0]=(1.5**beta)*((-beta**2)/3-beta+(23/6))-(.5**beta)*((-4*beta**2)/3-4*beta-2/3)
        B_R_Plus[0,0]=-B_R_Min[1,0]
        B_R_Min[2,0]=(.5**beta)*((-beta**2)/3-beta-(1/6))
        B_R_Plus[1,0]=-B_R_Min[2,0]
        constant=((1-self.gamma)*(h**(beta-1)))/(2*gamma(beta+3))
        K_plus_diag=constant*diag(K_m_1[1:],k=0)
        K_min_diag=constant*diag(K_m_1[:N],k=0)
        B=cp.matmul(K_plus_diag,B_R_Plus)+cp.matmul(K_min_diag,B_R_Min)
        mempool.free_all_blocks()
        return B
    def Cubic_Right_Test(self):
        beta=self.beta
        g=self.gamma
        N=self.N
        h=self.h[1]
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_1=cp.ones(self.N+1)*.01
        K_m=cp.zeros(self.N+1)
        # K_m=coeff(x=self.mid[0:self.N+1],t=t)
        k_left_non_lin=lambda x: 1+x
        K_m_ss_nonlin=k_left_non_lin(self.mid[0:])
        col_linspace=cp.linspace(3,self.N-1, self.N-3)
        j=cp.linspace(3,self.N, self.N-2)
        j_1=cp.linspace(4,self.N, self.N-3)
        j_2=cp.linspace(5,self.N, self.N-4)
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
        B_L_Min[:,0],B_L_Plus[:,0],B_L_Min[:,1],B_L_Plus[:,1]=0,0,0,0
        B_L_Min[0,0]=(-1)*((.5)**beta)*(6*(beta**2)+13*beta+3.5)
        B_L_Plus[0,0]=((1.5)**beta)*(6*(beta**2)+3*beta-4.5)
        B_L_Min[0,1]=((.5)**beta)*((3*beta**2)+5*beta-.5)
        B_L_Plus[0,1]=((1.5)**beta)*(-3*(beta**2)+3*beta+4.5)
        B_L_Min[1,0]=((1.5)**beta)*(-6*(beta**2)-3*beta+4.5)
        B_L_Plus[1,0]=((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((2.5)**beta)*(-6*(beta**2)+7*beta+.5)
        B_L_Min[1,1]=-B_L_Plus[0,1]
        B_L_Plus[1,1]=((.5)**beta)*(-2*(beta**2)-6*beta-1)-((2.5)**beta)*(3*(beta**2)-11*beta+3.5)
        B_L_Min[2,0]=-B_L_Plus[1,0]
        B_L_Min[2,1]=-B_L_Plus[1,1]
        B_L_Plus[2,1]=((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((3.5)**beta)*(-3*beta**2+19*beta-23.5)
        B_L_Min[3,1]=-B_L_Plus[2,1]
        B_L_Plus[0,2]=(-1)*((1.5)**beta)*(-2*(beta**2)/3+beta+(1/6))
        B_L_Min[0,2]=((.5)**beta)*(-2*(beta**2)/3-beta+(1/6))
        B_L_Min[1,2]=-B_L_Plus[0,2]
        B_L_Plus[1,2]=((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))-((2.5)**beta)*(-2*(beta**2)/3+3*beta-23/6)
        B_L_Min[2,2]=-B_L_Plus[1,2]
        B_L_Plus[2,2]=((1.5)**beta)*(4*(beta**2)/3+4*beta-(46/3))-((.5**beta)*(2*beta**2+6*beta+1)+(3.5**beta)*(-2*(beta**2)/3+5*beta-(71/6)))
        B_L_Min[3,2]=-B_L_Plus[2,2]
        B_L_Plus[3,2]=((2.5)**beta)*(4*(beta**2)/3+4*beta-(142/3))+((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))\
            -((1.5**beta)*(2*(beta**2)+6*beta-23)+(4.5**beta)*(-2*(beta**2)/3+7*beta-(143/6)))
        B_L_Min[4,2]=-B_L_Plus[3,2]
        B_L_Plus[4:,2]=((j_2-3.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+56*j_2-(286/3))+((j_2-1.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+24*j_2-(46/3))-(((j_2-4.5)**beta)*((beta**2)/3+beta-2*j_2**2+18*j_2-(239/6))+((j_2-2.5)**beta)*(2*beta**2+6*beta-12*j_2**2+60*j_2-71)+((j_2+.5)**beta)*(-2*(beta**2)/3-beta+2*beta*j_2+2*j_2-2*j_2**2+(1/6)))
        B_L_Min[5:,2]=-B_L_Plus[4:self.N-1,2]
        B_L_Plus[2:,0]=((j-1.5)**beta)*((4/3)*(beta**2)+4*beta-8*j**2+24*j-(46/3))-((j-2.5)**beta)*((1/3)*(beta**2)+beta-2*j**2+10*j-(71/6))-\
            ((j+.5)**beta)*(-6*(beta**2)-13*beta+10*beta*j-6*j**2+14*j-3.5)
        B_L_Min[3:,0]=-B_L_Plus[2:self.N-1,0]
        B_L_Plus[3:,1]=-((j_1-3.5)**beta)*((1/3)*(beta**2)+beta-2*(j_1**2)+14*j_1-(143/6))+((j_1-2.5)**beta)*((4/3)*beta**2+4*beta+40*j_1-8*(j_1**2)-(142/3))+\
            ((j_1+.5)**beta)*(-3*beta**2-5*beta+8*beta*(j_1)+10*j_1-6*(j_1**2)+.5)-((j_1-1.5)**beta)*(2*beta**2+6*beta-12*j_1**2+36*j_1-23)
        B_L_Min[4:,1]=-B_L_Plus[3:self.N-1,1]
        B_L_Plus[self.N-1,self.N-1]=((1.5)**beta)*((4/3)*beta**2+4*beta-(46/3))-((2.5)**beta)*((1/3)*beta**2+beta-(71/6))
        B_L_Min[self.N-1,self.N-1]=((1.5)**beta)*((1/3)*beta**2+beta-(23/6))-((.5)**beta)*((4/3)*beta**2+4*beta+(2/3))
        B_L_Plus[self.N-2,self.N-1]=-B_L_Min[self.N-1,self.N-1]
        B_L_Min[self.N-2,self.N-1]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Plus[self.N-3,self.N-1]=-B_L_Min[self.N-2,self.N-1]
        B_L_Plus[self.N-1,self.N-2]=-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((2.5)**beta)*((4/3)*(beta**2)+4*beta-(142/3))-\
            ((3.5)**beta)*((1/3)*(beta**2)+beta-(143/6))
        B_L_Min[self.N-1,self.N-2]=((.5)**beta)*(2*(beta**2)+6*beta+1)-((1.5)**beta)*((4/3)*(beta**2)+4*beta-(46/3))+\
            ((2.5)**beta)*((1/3)*(beta**2)+beta-(71/6))
        B_L_Plus[self.N-2,self.N-2]=-B_L_Min[self.N-1,self.N-2]
        B_L_Min[self.N-2,self.N-2]=((1.5)**beta)*((1/3)*(beta**2)+beta-(23/6))-((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))
        B_L_Plus[self.N-3,self.N-2]=-B_L_Min[self.N-2,self.N-2]
        B_L_Min[self.N-3,self.N-2]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Plus[self.N-4,self.N-2]=-B_L_Min[self.N-3,self.N-2]
        B_L_Plus[self.N-1,self.N-3]=((3.5)**beta)*(4*(beta**2)/3+4*beta-(286/3))+((1.5)**beta)*(4*(beta**2)/3+4*beta-46/3)-((4.5**beta)*((beta**2)/3+beta-(239/6))+(2.5**beta)*(2*(beta**2)+6*beta-71))
        B_R_Plus=cp.rot90(B_L_Min,2)
        B_R_Min=cp.rot90(B_L_Plus,2)
        print(B_R_Plus)
        constant=((1-self.gamma)*(h**(beta-1)))/(2*gamma(beta+3))
        K_plus_diag=constant*diag(K_m_ss_nonlin[1:],k=0)
        K_min_diag=constant*diag(K_m_ss_nonlin[:N],k=0)
        B=cp.matmul(K_plus_diag,B_R_Plus)+cp.matmul(K_min_diag,B_R_Min)
        return B
    def Linear_Left_Deriv(self):
        n=cp.linspace(2,self.N-1, self.N-2)
        # k_left_non_lin=lambda x: 1+x**3
        k_left_non_lin=lambda x: (1+x)**2
        K_m_ss=cp.ones(self.N+1)*.01
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
        K_plus_diag=const*diag(K_m_ss_nonlin[1:],k=0)
        K_min_diag=const*diag(K_m_ss_nonlin[:self.N],k=0)
        B_L_Plus=toeplitz(c=col_plus,r=row_plus)
        B_L_Min=toeplitz(c=col_min,r=row_min)
        B=cp.matmul(K_plus_diag,B_L_Plus)+cp.matmul(K_min_diag,B_L_Min)
        return B
    def Linear_Right_Deriv(self):
        n=cp.linspace(2,self.N-1, self.N-2)
        k_left_non_lin=lambda x: 1+x
        K_m_ss=cp.ones(self.N+1)*.01
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
        B_L_Plus=toeplitz(c=col_plus,r=row_plus)
        B_L_Min=toeplitz(c=col_min,r=row_min)
        const=(1-self.gamma)/(gamma(self.beta+1)*self.h[0]**(1-self.beta))
        K_p,K_m=cp.empty(shape=(self.N,self.N)),cp.empty(shape=(self.N,self.N))
        K_p=const*diag(K_m_ss[1:],k=0)
        K_m=const*diag(K_m_ss[:self.N],k=0)
        B_R_Plus=cp.rot90(B_L_Min,2)
        B_R_Min=cp.rot90(B_L_Plus,2)
        B=cp.matmul(K_p,B_R_Plus)+cp.matmul(K_m,B_R_Min)
        return B
    def Mem_Effic_B_Left(self):
        beta=self.beta
        g=self.gamma
        h=self.h[1]
        coeff=lambda x,t: .002*(1+x*(2-x)+t**2)
        K_m_1=cp.ones(self.N+1)*.01
        K_m=cp.zeros(self.N+1)
        #k_left_non_lin=lambda x: 1+x**3
        k_left_non_lin=lambda x: (1+x)**2
        K_m_ss_nonlin=k_left_non_lin(self.mid[0:])
        col_linspace=cp.linspace(3,self.N, self.N-2)
        j=cp.linspace(3,self.N, self.N-2)
        j_1=cp.linspace(4,self.N, self.N-3)
        j_2=cp.linspace(5,self.N, self.N-4)
        col_min=cp.empty(shape=self.N+1)
        col_min[3:]=((col_linspace+1.5)**beta)*(-2*(col_linspace**2)-6*col_linspace+((beta**2)/3)+beta-(23/6))+(2/3)*((col_linspace+.5)**beta)*(12*(col_linspace**2)+12*col_linspace-2*(beta**2)-6*beta-1)+\
            ((col_linspace-.5)**beta)*(-12*(col_linspace**2)+12*col_linspace+2*(beta**2)+6*beta+1)+(2/3)*((col_linspace-1.5)**beta)*(12*(col_linspace**2)-36*col_linspace-2*beta**2-6*beta+23)+\
            ((col_linspace-2.5)**beta)*(-2*(col_linspace**2)+10*col_linspace+((beta**2)/3)+beta-(71/6))
        col_min[2]=((3.5**beta))*(((beta**2)/3)+beta-(143/6))+((2.5)**beta)*((-4*(beta**2)/3)-4*beta+(142/3))+((1.5)**beta)*(2*(beta**2)+6*beta-23)+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
        col_min[1]=((2.5)**beta)*(((beta**2)/3)+beta-(71/6))+((1.5)**beta)*((-4*(beta**2)/3)-4*beta+(46/3))+((.5)**beta)*(2*(beta**2)+6*beta+1)
        col_min[0]=((1.5)**beta)*(((beta**2)/3)+beta-(23/6))+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
        row_min=cp.zeros(shape=self.N)
        row_min[0]=((1.5)**beta)*(((beta**2)/3)+beta-(23/6))+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
        row_min[1]=((.5)**beta)*(((beta**2)/3)+beta+(1/6))
        B_L_Min=toeplitz(c=col_min,r=row_min)
        B_L_Min[0,0]=(-1)*((.5)**beta)*(6*(beta**2)+13*beta+3.5)
        B_L_Min[0,1]=((.5)**beta)*((3*beta**2)+5*beta-.5)
        B_L_Min[1,0]=((1.5)**beta)*(-6*(beta**2)-3*beta+4.5)
        B_L_Min[1,1]=-(((1.5)**beta)*(-3*(beta**2)+3*beta+4.5))
        B_L_Min[2,0]=-(((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((2.5)**beta)*(-6*(beta**2)+7*beta+.5))
        B_L_Min[2,1]=-(((.5)**beta)*(-2*(beta**2)-6*beta-1)-((2.5)**beta)*(3*(beta**2)-11*beta+3.5))
        B_L_Min[3,1]=-(((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((3.5)**beta)*(-3*beta**2+19*beta-23.5))
        B_L_Min[0,2]=((.5)**beta)*(-2*(beta**2)/3-beta+(1/6))
        B_L_Min[1,2]=((1.5)**beta)*(-2*(beta**2)/3+beta+(1/6))
        B_L_Min[2,2]=-(((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))-((2.5)**beta)*(-2*(beta**2)/3+3*beta-23/6))
        B_L_Min[3,2]=-(((1.5)**beta)*(4*(beta**2)/3+4*beta-(46/3))-((.5**beta)*(2*beta**2+6*beta+1)+(3.5**beta)*(-2*(beta**2)/3+5*beta-(71/6))))
        B_L_Min[4,2]=-(((2.5)**beta)*(4*(beta**2)/3+4*beta-(142/3))+((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))\
            -((1.5**beta)*(2*(beta**2)+6*beta-23)+(4.5**beta)*(-2*(beta**2)/3+7*beta-(143/6))))
        B_L_Min[5:,2]=-(((j_2-3.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+56*j_2-(286/3))+((j_2-1.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+24*j_2-(46/3))-(((j_2-4.5)**beta)*((beta**2)/3+beta-2*j_2**2+18*j_2-(239/6))+((j_2-2.5)**beta)*(2*beta**2+6*beta-12*j_2**2+60*j_2-71)+((j_2+.5)**beta)*(-2*(beta**2)/3-beta+2*beta*j_2+2*j_2-2*j_2**2+(1/6))))
        B_L_Min[3:,0]=-(((j-1.5)**beta)*((4/3)*(beta**2)+4*beta-8*j**2+24*j-(46/3))-((j-2.5)**beta)*((1/3)*(beta**2)+beta-2*j**2+10*j-(71/6))-\
            ((j+.5)**beta)*(-6*(beta**2)-13*beta+10*beta*j-6*j**2+14*j-3.5))
        B_L_Min[4:,1]=-(-((j_1-3.5)**beta)*((1/3)*(beta**2)+beta-2*(j_1**2)+14*j_1-(143/6))+((j_1-2.5)**beta)*((4/3)*beta**2+4*beta+40*j_1-8*(j_1**2)-(142/3))+\
            ((j_1+.5)**beta)*(-3*beta**2-5*beta+8*beta*(j_1)+10*j_1-6*(j_1**2)+.5)-((j_1-1.5)**beta)*(2*beta**2+6*beta-12*j_1**2+36*j_1-23))
        B_L_Min[self.N,self.N-1]=-(((1.5)**beta)*((4/3)*beta**2+4*beta-(46/3))-((2.5)**beta)*((1/3)*beta**2+beta-(71/6)))
        B_L_Min[self.N-1,self.N-1]=((1.5)**beta)*((1/3)*beta**2+beta-(23/6))-((.5)**beta)*((4/3)*beta**2+4*beta+(2/3))
        B_L_Min[self.N-2,self.N-1]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Min[self.N,self.N-2]=-(-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((2.5)**beta)*((4/3)*(beta**2)+4*beta-(142/3))-\
            ((3.5)**beta)*((1/3)*(beta**2)+beta-(143/6)))
        B_L_Min[self.N-1,self.N-2]=((.5)**beta)*(2*(beta**2)+6*beta+1)-((1.5)**beta)*((4/3)*(beta**2)+4*beta-(46/3))+\
            ((2.5)**beta)*((1/3)*(beta**2)+beta-(71/6))
        B_L_Min[self.N-2,self.N-2]=((1.5)**beta)*((1/3)*(beta**2)+beta-(23/6))-((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))
        B_L_Min[self.N-3,self.N-2]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
        B_L_Min[self.N,self.N-3]=-(((3.5)**beta)*(4*(beta**2)/3+4*beta-(286/3))+((1.5)**beta)*(4*(beta**2)/3+4*beta-46/3)-((4.5**beta)*((beta**2)/3+beta-(239/6))+(2.5**beta)*(2*(beta**2)+6*beta-71)))
        constant=(self.gamma*(h**(beta-1)))/(2*gamma(beta+3))
        K_plus_diag=constant*diag(K_m_ss_nonlin[1:],k=0)
        K_min_diag=constant*diag(K_m_ss_nonlin[:self.N],k=0)
        B=-cp.matmul(K_plus_diag,B_L_Min[1:,:])+cp.matmul(K_min_diag,B_L_Min[:-1,:])
        return B



