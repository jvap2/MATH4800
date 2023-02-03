import cupy as cp
from cupy import diag
import cupyx.scipy
from cupyx.scipy.linalg import toeplitz
from cupyx.scipy.sparse import diags
from cupyx.scipy.sparse.linalg import aslinearoperator
from math import gamma
from cupyx.scipy.sparse.linalg import LinearOperator
from d_mesh import Mesh
import math
import d_mesh
from d_mesh import Mesh
from d_stiff_matrix import StiffMatrix

def B_1_Cubic_Right_Min(N,beta):
    B_R_Min=cp.zeros(shape=(N,3))
    B_R_Min[0,2]=(4.5**beta)*((-beta**2)/3-beta+(239/6))+(2.5**beta)*(-2*beta**2-6*beta+71)-((3.5**beta)*((-4*beta**2)/3-4*beta+(286/3))+(1.5**beta)*((-4*beta**2)/3-4*beta+(46/3)))
    B_R_Min[0,1]=(3.5**beta)*((-beta**2)/3-beta+143/6)+(1.5**beta)*(-2*beta**2-6*beta+23)-(2.5**beta)*((-4*beta**2)/3-4*beta+142/3)
    B_R_Min[1,1]=(2.5**beta)*((-beta**2)/3-beta+(71/6))+(.5**beta)*(-2*beta**2-6*beta-1)-(1.5**beta)*((-4*beta**2)/3-4*beta+46/3)
    B_R_Min[2,1]=(1.5**beta)*((-beta**2)/3-beta+(23/6))-(.5**beta)*((-4*beta**2)/3-4*beta-(2/3))
    B_R_Min[3,1]=(.5**beta)*((-beta**2)/3-beta-(1/6))
    B_R_Min[0,0]=(2.5**beta)*((-beta**2)/3-beta+(71/6))-(1.5**beta)*((-4*beta**2)/3-4*beta+(46/3))
    B_R_Min[1,0]=(1.5**beta)*((-beta**2)/3-beta+(23/6))-(.5**beta)*((-4*beta**2)/3-4*beta-2/3)
    B_R_Min[2,0]=(.5**beta)*((-beta**2)/3-beta-(1/6))
    return aslinearoperator(B_R_Min)

def B_1_Cubic_Right_Plus(N,beta):
    B_R_Plus=cp.zeros(shape=(N,2))  
    B_R_Plus[0,1]=-(2.5**beta)*((-beta**2)/3-beta+(71/6))-(.5**beta)*(-2*beta**2-6*beta-1)+(1.5**beta)*((-4*beta**2)/3-4*beta+46/3)
    B_R_Plus[1,1]=-(1.5**beta)*((-beta**2)/3-beta+(23/6))+(.5**beta)*((-4*beta**2)/3-4*beta-(2/3))
    B_R_Plus[2,1]=-(.5**beta)*((-beta**2)/3-beta-(1/6))
    B_R_Plus[0,0]=-(1.5**beta)*((-beta**2)/3-beta+(23/6))+(.5**beta)*((-4*beta**2)/3-4*beta-2/3)
    B_R_Plus[1,0]=-(.5**beta)*((-beta**2)/3-beta-(1/6))
    return aslinearoperator(B_R_Plus)

def B_3_Cubic_Right_Min(N,beta):
    j_1=cp.linspace(1,N-2,N-2)
    j=cp.linspace(1,N-3, N-3)
    B_R_Min=cp.zeros(shape=(N,3))
    B_R_Min[0:N-2,2]=((N-j_1-1.5)**beta)*((-beta**2)/3-beta+2*j_1**2+j_1*(6-4*N)+2*N**2-6*N+23/6)-((N-j_1-.5)**beta)*((-4*beta**2)/3-4*beta+8*j_1**2+(8-16*N)*j_1+8*N**2-8*N-2/3)+\
    ((N-j_1+1.5)**beta)*(6*beta**2+(3-10*N)*beta+10*beta*j_1+6*j_1**2+(2-12*N)*j_1+6*N**2-2*N-4.5)
    B_R_Min[N-2,2]=((2.5)**beta)*(6*beta**2-7*beta-.5)-((.5)**beta)*((-4*beta**2)/3-4*beta-2/3)
    B_R_Min[N-1,2]=((1.5)**beta)*(6*beta**2+3*beta-4.5)
    B_R_Min[0:N-3,1]=((N-j-2.5)**beta)*(-(beta**2)/3-beta+2*j**2+j*(10-4*N)+2*N**2-10*N+71/6)-((N-j-1.5)**beta)*((-4*beta**2)/3-4*beta+8*j**2+24*j-16*j*N+8*N**2-24*N+46/3)+\
        ((N-j-.5)**beta)*(-2*beta**2-6*beta+12*j**2+12*j-24*N*j+12*N**2-12*N-1)-((N-j+1.5)**beta)*(3*beta**2+beta*(8*j-8*N-3)+6*j**2-2*j*(6*N+1)+6*N**2+2*N-4.5)
    B_R_Min[N-1,1]=(-(1.5)**beta)*(3*beta**2-3*beta-4.5)
    B_R_Min[N-2,1]=((.5)**beta)*(-2*beta**2-6*beta-1)-(2.5**beta)*(3*beta**2-11*beta+3.5)
    B_R_Min[N-3,1]=(-(.5)**beta)*((-4*beta**2)/3-4*beta-2/3)+((1.5)**beta)*(-2*beta**2-6*beta+23)-(3.5**beta)*(3*beta**2-19*beta+23.5)
    B_R_Min[0:N-4,0]=-((N-j[:-1]-2.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-1]**2+40*j[:-1]-16*j[:-1]*N+8*N**2-40*N+142/3)+((N-j[:-1]-3.5)**beta)*((-beta**2)/3-beta+2*j[:-1]**2+j[:-1]*(14-4*N)+2*N**2-14*N+143/6)+\
        ((N-j[:-1]-1.5)**beta)*(-2*beta**2-6*beta+12*j[:-1]**2+36*j[:-1]-24*j[:-1]*N+12*N**2-36*N+23)-((N-j[:-1]-.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-1]**2+8*j[:-1]-16*j[:-1]*N+8*N**2-8*N-2/3)+\
        ((N-j[:-1]+1.5)**beta)*((2*beta**2)/3+beta*(2*j[:-1]-2*N-1)+2*j[:-1]**2+j[:-1]*(-4*N-2)+2*N**2+2*N-1/6)
    B_R_Min[N-1,0]=((1.5)**beta)*((2*beta**2)/3-beta-1/6)
    B_R_Min[N-2,0]=-((.5)**beta)*((-4*beta**2)/3-4*beta-2/3)+((2.5)**beta)*((2*beta**2)/3-3*beta+23/6)
    B_R_Min[N-3,0]=(.5**beta)*(-2*(beta**2)-6*beta-1)-(1.5**beta)*((-4*beta**2)/3-4*beta+46/3)+(3.5**beta)*((2*beta**2)/3-5*beta+71/6)
    B_R_Min[N-4,0]=-((.5)**beta)*((-4*beta**2)/3-4*beta-(2/3))+(1.5**beta)*(-2*beta**2-6*beta+23)-(2.5**beta)*((-4*beta**2)/3-4*beta+142/3)+(4.5**beta)*((2*beta**2)/3-7*beta+143/6)
    return aslinearoperator(B_R_Min)



def B_3_Cubic_Right_Plus(N,beta):
    j=cp.linspace(1,N-3, N-3)
    B_R_Plus=cp.zeros(shape=(N,3))
    B_R_Plus[0:N-3,2]=(-(N-j+.5)**beta)*(6*(beta**2)+(13-10*N)*beta+10*beta*j+6*(j**2)+j*(14-12*N)+6*(N**2)-14*N+3.5)+\
        ((N-j-1.5)**beta)*((-4*beta**2)/3-4*beta+8*(j**2)+j*(24-16*N)+8*(N**2)-24*N+(46/3))-((N-j-2.5)**beta)*((-beta**2)/3-beta+2*(j**2)+j*(10-4*N)+2*(N**2)-10*N+71/6)
    B_R_Plus[N-3,2]=(-(2.5)**beta)*(6*beta**2-7*beta-.5)+((.5)**beta)*((-4*beta**2)/3-4*beta-2/3)
    B_R_Plus[N-2,2]=(-(1.5)**beta)*(6*beta**2+3*beta-4.5)
    B_R_Plus[N-1,2]=(-(.5)**beta)*(6*beta**2+13*beta+3.5)
    B_R_Plus[0:N-4,1]=-((N-j[:-1]-3.5)**beta)*((-beta**2)/3-beta+2*j[:-1]**2+j[:-1]*(14-4*N)+2*N**2-14*N+143/6)+((N-j[:-1]-2.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-1]**2+40*j[:-1]-16*N*j[:-1]+8*N**2-40*N+142/3)-\
        ((N-j[:-1]-1.5)**beta)*(-2*beta**2-6*beta+12*j[:-1]**2+36*j[:-1]-24*j[:-1]*N+12*N**2-36*N+23)+((N-j[:-1]+.5)**beta)*(3*beta**2+beta*(8*j[:-1]-8*N+5)+6*j[:-1]**2+j[:-1]*(10-12*N)+6*N**2-10*N-.5)
    B_R_Plus[N-1,1]=((.5)**beta)*(3*beta**2+5*beta-.5)
    B_R_Plus[N-2,1]=((1.5)**beta)*(3*beta**2-3*beta-4.5)
    B_R_Plus[N-3,1]=-((.5)**beta)*(-2*beta**2-6*beta-1+2.5**beta)*(3*beta**2-11*beta+3.5)
    B_R_Plus[N-4,1]=((.5)**beta)*((-4*beta**2)/3-4*beta-2/3)-((1.5)**beta)*(-2*beta**2-6*beta+23)+(3.5**beta)*(3*beta**2-19*beta+23.5)
    B_R_Plus[0:N-5,0]=((N-j[:-2]-3.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-2]**2-16*j[:-2]*N+56*j[:-2]+8*N**2-56*N+286/3)-((N-j[:-2]-4.5)**beta)*((-beta**2)/3-beta+2*j[:-2]**2+j[:-2]*(18-4*N)+2*N**2-18*N+239/6)-\
        ((N-j[:-2]-2.5)**beta)*(-2*beta**2-6*beta+12*j[:-2]**2+60*j[:-2]-24*j[:-2]*N+12*N**2-60*N+71)+((N-j[:-2]-1.5)**beta)*((-4*beta**2)/3-4*beta+8*j[:-2]**2+24*j[:-2]-16*N*j[:-2]+8*N**2-24*N+46/3)-\
        ((N-j[:-2]+.5)**beta)*((2*beta**2)/3+beta*(2*j[:-2]-2*N+1)+2*j[:-2]**2+j[:-2]*(2-4*N)+2*N**2-2*N-1/6)
    B_R_Plus[N-1,0]=-((.5)**beta)*((2*beta**2)/3+beta-1/6)
    B_R_Plus[N-2,0]=-((1.5)**beta)*((2*beta**2)/3-beta-1/6)
    B_R_Plus[N-3,0]=((.5)**beta)*((-4*beta**2)/3-4*beta-2/3)-((2.5)**beta)*((2*beta**2)/3-3*beta+23/6)
    B_R_Plus[N-4,0]=-(.5**beta)*(-2*(beta**2)-6*beta-1)+(1.5**beta)*((-4*beta**2)/3-4*beta+46/3)-(3.5**beta)*((2*beta**2)/3-5*beta+71/6)
    B_R_Plus[N-5,0]=((.5)**beta)*((-4*beta**2)/3-4*beta-(2/3))-(1.5**beta)*(-2*beta**2-6*beta+23)+(2.5**beta)*((-4*beta**2)/3-4*beta+142/3)-(4.5**beta)*((2*beta**2)/3-7*beta+143/6)
    return aslinearoperator(B_R_Plus)
    
def B_2_Cubic_Right_Plus(N,beta):
    row_plus=cp.zeros(shape=N-5)
    col_plus=cp.zeros(shape=N)
    q=cp.linspace(-2,-(N-4), N-5)
    row_plus[1:]=((-q[1:]+.5)**beta)*((-4*beta**2)/3-4*beta+8*q[1:]**2-8*q[1:]-(2/3))+((-q[1:]-1.5)**beta)*((-4*beta**2)/3-4*beta+8*q[1:]**2+24*q[1:]+(46/3))-\
    (((-q[1:]+1.5)**beta)*((-beta**2)/3-beta+2*q[1:]**2-6*q[1:]+(23/6))+((-q[1:]-.5)**beta)*(-2*beta**2-6*beta+12*q[1:]**2+12*q[1:]-1)+((-q[1:]-2.5)**beta)*((-beta**2)/3-beta+2*q[1:]**2+10*q[1:]+(71/6)))
    row_plus[0]=(2.5**beta)*((-4*beta**2)/3-4*beta+(142/3))+(.5**beta)*((-4*beta**2)/3-4*beta-(2/3))-((3.5**beta)*((-beta**2)/3-beta+(143/6))+(1.5**beta)*(-2*beta**2-6*beta+23))
    col_plus[0]=row_plus[0]
    col_plus[1]=(1.5**beta)*((-4*beta**2)/3-4*beta+(46/3))-((2.5**beta)*((-beta**2)/3-beta+71/6)+(.5**beta)*(-2*beta**2-6*beta-1))
    col_plus[2]=(.5**beta)*((-4*beta**2)/3-4*beta-2/3)-(1.5**beta)*((-beta**2)/3-beta+23/6)
    col_plus[3]=(-1)*((.5)**beta)*((-beta**2)/3-beta-1/6)
    B_R_Plus=toeplitz(c=col_plus,r=row_plus)
    return aslinearoperator(B_R_Plus)


def B_2_Cubic_Right_Min(N,beta):
    row_min=cp.zeros(shape=N-5)
    col_min=cp.zeros(shape=N)
    q=cp.linspace(-2,-(N-4), N-5)
    row_min=((-q+2.5)**beta)*((-beta**2)/3-beta+2*q**2-10*q+(71/6))+((-q+.5)**beta)*(-2*beta**2-6*beta+12*q**2-12*q-1)+((-q-1.5)**beta)*((-beta**2)/3-beta+2*q**2+6*q+(23/6))-\
    (((-q+1.5)**beta)*((-4*beta**2)/3-4*beta+8*q**2-24*q+(46/3))+((-q-.5)**beta)*((-4*beta**2)/3-4*beta+8*q**2+8*q-(2/3)))
    col_min[0]=row_min[0]
    col_min[1]=-(2.5**beta)*((-4*beta**2)/3-4*beta+(142/3))-(.5**beta)*((-4*beta**2)/3-4*beta-(2/3))+((3.5**beta)*((-beta**2)/3-beta+(143/6))+(1.5**beta)*(-2*beta**2-6*beta+23))
    col_min[2]=-(1.5**beta)*((-4*beta**2)/3-4*beta+(46/3))+((2.5**beta)*((-beta**2)/3-beta+71/6)+(.5**beta)*(-2*beta**2-6*beta-1))
    col_min[3]=-(.5**beta)*((-4*beta**2)/3-4*beta-2/3)+(1.5**beta)*((-beta**2)/3-beta+23/6)
    col_min[4]=((.5)**beta)*((-beta**2)/3-beta-1/6)
    B_R_Min=toeplitz(c=col_min,r=row_min)
    return aslinearoperator(B_R_Min)



def B_1_Cubic_Left_Plus(N,beta):
    B_L_Plus=cp.zeros(shape=(N,3))
    j=cp.linspace(3,N,N-2)
    j_1=cp.linspace(4,N,N-3)
    j_2=cp.linspace(5,N,N-4)
    B_L_Plus[0,0]=((1.5)**beta)*(6*(beta**2)+3*beta-4.5)
    B_L_Plus[0,1]=((1.5)**beta)*(-3*(beta**2)+3*beta+4.5)
    B_L_Plus[1,0]=((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((2.5)**beta)*(-6*(beta**2)+7*beta+.5)
    B_L_Plus[1,1]=((.5)**beta)*(-2*(beta**2)-6*beta-1)-((2.5)**beta)*(3*(beta**2)-11*beta+3.5)
    B_L_Plus[2,1]=((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((3.5)**beta)*(-3*beta**2+19*beta-23.5)
    B_L_Plus[0,2]=(-1)*((1.5)**beta)*(-2*(beta**2)/3+beta+(1/6))
    B_L_Plus[1,2]=((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))-((2.5)**beta)*(-2*(beta**2)/3+3*beta-23/6)
    B_L_Plus[2,2]=((1.5)**beta)*(4*(beta**2)/3+4*beta-(46/3))-((.5**beta)*(2*beta**2+6*beta+1)+(3.5**beta)*(-2*(beta**2)/3+5*beta-(71/6)))
    B_L_Plus[3,2]=((2.5)**beta)*(4*(beta**2)/3+4*beta-(142/3))+((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))\
            -((1.5**beta)*(2*(beta**2)+6*beta-23)+(4.5**beta)*(-2*(beta**2)/3+7*beta-(143/6)))
    B_L_Plus[4:,2]=((j_2-3.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+56*j_2-(286/3))+((j_2-1.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+24*j_2-(46/3))-(((j_2-4.5)**beta)*((beta**2)/3+beta-2*j_2**2+18*j_2-(239/6))+((j_2-2.5)**beta)*(2*beta**2+6*beta-12*j_2**2+60*j_2-71)+((j_2+.5)**beta)*(-2*(beta**2)/3-beta+2*beta*j_2+2*j_2-2*j_2**2+(1/6)))
    B_L_Plus[2:,0]=((j-1.5)**beta)*((4/3)*(beta**2)+4*beta-8*j**2+24*j-(46/3))-((j-2.5)**beta)*((1/3)*(beta**2)+beta-2*j**2+10*j-(71/6))-\
            ((j+.5)**beta)*(-6*(beta**2)-13*beta+10*beta*j-6*j**2+14*j-3.5)
    B_L_Plus[3:,1]=-((j_1-3.5)**beta)*((1/3)*(beta**2)+beta-2*(j_1**2)+14*j_1-(143/6))+((j_1-2.5)**beta)*((4/3)*beta**2+4*beta+40*j_1-8*(j_1**2)-(142/3))+\
            ((j_1+.5)**beta)*(-3*beta**2-5*beta+8*beta*(j_1)+10*j_1-6*(j_1**2)+.5)-((j_1-1.5)**beta)*(2*beta**2+6*beta-12*j_1**2+36*j_1-23)
    return aslinearoperator(B_L_Plus)
    

def B_1_Cubic_Left_Min(N,beta):
    j_2=cp.linspace(5,N-1, N-5)
    j=cp.linspace(3,N-1, N-3)
    j_1=cp.linspace(4,N-1,N-4)
    B_L_Min=cp.zeros(shape=(N,3))
    B_L_Min[0,0]=(-1)*((.5)**beta)*(6*(beta**2)+13*beta+3.5)
    B_L_Min[0,1]=((.5)**beta)*((3*beta**2)+5*beta-.5)
    B_L_Min[1,0]=((1.5)**beta)*(-6*(beta**2)-3*beta+4.5)
    B_L_Min[1,1]=(-1)*((1.5)**beta)*(-3*(beta**2)+3*beta+4.5)
    B_L_Min[2,0]=(-1)*(((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((2.5)**beta)*(-6*(beta**2)+7*beta+.5))
    B_L_Min[2,1]=(-1)*(((.5)**beta)*(-2*(beta**2)-6*beta-1)-((2.5)**beta)*(3*(beta**2)-11*beta+3.5))
    B_L_Min[3,1]=(-1)*(((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((3.5)**beta)*(-3*beta**2+19*beta-23.5))
    B_L_Min[0,2]=((.5)**beta)*(-2*(beta**2)/3-beta+(1/6))
    B_L_Min[1,2]=((1.5)**beta)*(-2*(beta**2)/3+beta+(1/6))
    B_L_Min[2,2]=(-1)*(((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))-((2.5)**beta)*(-2*(beta**2)/3+3*beta-23/6))
    B_L_Min[3,2]=(-1)*(((1.5)**beta)*(4*(beta**2)/3+4*beta-(46/3))-((.5**beta)*(2*beta**2+6*beta+1)+(3.5**beta)*(-2*(beta**2)/3+5*beta-(71/6))))
    B_L_Min[4,2]=(-1)*(((2.5)**beta)*(4*(beta**2)/3+4*beta-(142/3))+((.5)**beta)*(4*(beta**2)/3+4*beta+(2/3))\
            -((1.5**beta)*(2*(beta**2)+6*beta-23)+(4.5**beta)*(-2*(beta**2)/3+7*beta-(143/6))))
    B_L_Min[5:,2]=(-1)*(((j_2-3.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+56*j_2-(286/3))+((j_2-1.5)**beta)*(4*(beta**2)/3+4*beta-8*j_2**2+24*j_2-(46/3))-(((j_2-4.5)**beta)*((beta**2)/3+beta-2*j_2**2+18*j_2-(239/6))+((j_2-2.5)**beta)*(2*beta**2+6*beta-12*j_2**2+60*j_2-71)+((j_2+.5)**beta)*(-2*(beta**2)/3-beta+2*beta*j_2+2*j_2-2*j_2**2+(1/6))))
    B_L_Min[3:,0]=(-1)*(((j-1.5)**beta)*((4/3)*(beta**2)+4*beta-8*j**2+24*j-(46/3))-((j-2.5)**beta)*((1/3)*(beta**2)+beta-2*j**2+10*j-(71/6))-\
            ((j+.5)**beta)*(-6*(beta**2)-13*beta+10*beta*j-6*j**2+14*j-3.5))
    B_L_Min[4:,1]=(-1)*(-((j_1-3.5)**beta)*((1/3)*(beta**2)+beta-2*(j_1**2)+14*j_1-(143/6))+((j_1-2.5)**beta)*((4/3)*beta**2+4*beta+40*j_1-8*(j_1**2)-(142/3))+\
            ((j_1+.5)**beta)*(-3*beta**2-5*beta+8*beta*(j_1)+10*j_1-6*(j_1**2)+.5)-((j_1-1.5)**beta)*(2*beta**2+6*beta-12*j_1**2+36*j_1-23))
    return aslinearoperator(B_L_Min)

    
def B_2_Cubic_Left_Plus(N,beta):
    col_plus=cp.zeros(shape=N)
    row_plus=cp.zeros(shape=N-5)
    col_linspace=cp.linspace(3,N-4,N-6)
    col_plus[1]=((.5)**beta)*((-(beta**2)/3)-beta-(1/6))
    col_plus[2]=((.5)**beta)*((4*(beta**2)/3)+4*beta+(2/3))+((1.5)**beta)*((-(beta**2)/3)-beta+(23/6))
    col_plus[3]=((2.5)**beta)*((-(beta**2)/3)-beta+(71/6))+((1.5)**beta)*((4*(beta**2)/3)+4*beta-(46/3))+((.5)**beta)*(-2*(beta**2)-6*beta-1)
    col_plus[4]=((3.5)**beta)*((-(beta**2)/3)-beta+(143/6))+((2.5)**beta)*((4*(beta**2)/3)+4*beta-(142/3))+((1.5)**beta)*(-2*(beta**2)-6*beta+23)+((.5)**beta)*((4*(beta**2)/3)+4*beta+(2/3))
    col_plus[5]=((4.5)**beta)*((239/6)-((beta**2)/3)-beta)+(2/3)*((3.5)**beta)*(-143+2*(beta**2)+6*beta)+\
            ((2.5)**beta)*(71-2*(beta**2)-6*beta)+(2/3)*((1.5)**beta)*(-23+2*(beta**2)+6*beta)+\
            ((.5)**beta)*(-(1/6)-((beta**2)/3)-beta)
    col_plus[6:]=((col_linspace+2.5)**beta)*(2*(col_linspace**2)+10*col_linspace-(beta**2/3)-beta+(71/6))+(2/3)*((col_linspace+1.5)**beta)*(-12*(col_linspace**2)-36*col_linspace+2*beta**2+6*beta-23)+\
            ((col_linspace+.5)**beta)*(12*(col_linspace**2)+12*col_linspace-2*beta**2-6*beta-1)+(2/3)*((col_linspace-.5)**beta)*(-12*(col_linspace**2)+12*col_linspace+2*beta**2+6*beta+1)+\
            ((col_linspace-1.5)**beta)*(2*(col_linspace**2)-6*col_linspace-((beta**2)/3)-beta+(23/6))
    print(col_linspace)
    B_L_Plus=toeplitz(c=col_plus,r=row_plus)
    return aslinearoperator(B_L_Plus)



def B_2_Cubic_Left_Min(N,beta):
    row_min=cp.zeros(shape=N-5)
    col_min=cp.zeros(shape=N)
    col_linspace=cp.linspace(3,N-4,N-6)
    col_min[2]=((.5)**beta)*(((beta**2)/3)+beta+(1/6))
    col_min[3]=((1.5)**beta)*(((beta**2)/3)+beta-(23/6))+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
    col_min[4]=((2.5)**beta)*(((beta**2)/3)+beta-(71/6))+((1.5)**beta)*((-4*(beta**2)/3)-4*beta+(46/3))+((.5)**beta)*(2*(beta**2)+6*beta+1)
    col_min[5]=((3.5**beta))*(((beta**2)/3)+beta-(143/6))+((2.5)**beta)*((-4*(beta**2)/3)-4*beta+(142/3))+((1.5)**beta)*(2*(beta**2)+6*beta-23)+((.5)**beta)*((-4*(beta**2)/3)-4*beta-(2/3))
    col_min[6:]=((col_linspace+1.5)**beta)*(-2*(col_linspace**2)-6*col_linspace+((beta**2)/3)+beta-(23/6))+(2/3)*((col_linspace+.5)**beta)*(12*(col_linspace**2)+12*col_linspace-2*(beta**2)-6*beta-1)+\
            ((col_linspace-.5)**beta)*(-12*(col_linspace**2)+12*col_linspace+2*(beta**2)+6*beta+1)+(2/3)*((col_linspace-1.5)**beta)*(12*(col_linspace**2)-36*col_linspace-2*beta**2-6*beta+23)+\
            ((col_linspace-2.5)**beta)*(-2*(col_linspace**2)+10*col_linspace+((beta**2)/3)+beta-(71/6))
    B_L_Min=toeplitz(c=col_min,r=row_min)
    return aslinearoperator(B_L_Min)



def B_3_Cubic_Left_Plus(N,beta):
    B_L_Plus=cp.zeros(shape=(N,3))
    B_L_Plus[N-1,2]=((1.5)**beta)*((4/3)*beta**2+4*beta-(46/3))-((2.5)**beta)*((1/3)*beta**2+beta-(71/6))
    B_L_Plus[N-2,2]=(-1)*(((1.5)**beta)*((1/3)*beta**2+beta-(23/6))-((.5)**beta)*((4/3)*beta**2+4*beta+(2/3)))
    B_L_Plus[N-3,2]=(-1)*(((.5)**beta)*((1/3)*(beta**2)+beta+(1/6)))
    B_L_Plus[N-1,1]=-((1.5)**beta)*(2*(beta**2)+6*beta-23)+((2.5)**beta)*((4/3)*(beta**2)+4*beta-(142/3))-\
        ((3.5)**beta)*((1/3)*(beta**2)+beta-(143/6))
    B_L_Plus[N-2,1]=(-1)*(((.5)**beta)*(2*(beta**2)+6*beta+1)-((1.5)**beta)*((4/3)*(beta**2)+4*beta-(46/3))+\
            ((2.5)**beta)*((1/3)*(beta**2)+beta-(71/6)))
    B_L_Plus[N-3,1]=(-1)*(((1.5)**beta)*((1/3)*(beta**2)+beta-(23/6))-((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3)))
    B_L_Plus[N-4,1]=(-1)*(((.5)**beta)*((1/3)*(beta**2)+beta+(1/6)))
    B_L_Plus[N-1,0]=((3.5)**beta)*(4*(beta**2)/3+4*beta-(286/3))+((1.5)**beta)*(4*(beta**2)/3+4*beta-46/3)-((4.5**beta)*((beta**2)/3+beta-(239/6))+(2.5**beta)*(2*(beta**2)+6*beta-71))
    return aslinearoperator(B_L_Plus)

def B_3_Cubic_Left_Min(N,beta):
    B_L_Min=cp.zeros(shape=(N,2))
    B_L_Min[N-1,1]=((1.5)**beta)*((1/3)*beta**2+beta-(23/6))-((.5)**beta)*((4/3)*beta**2+4*beta+(2/3))
    B_L_Min[N-2,1]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
    B_L_Min[N-1,0]=((.5)**beta)*(2*(beta**2)+6*beta+1)-((1.5)**beta)*((4/3)*(beta**2)+4*beta-(46/3))+\
            ((2.5)**beta)*((1/3)*(beta**2)+beta-(71/6))
    B_L_Min[N-2,0]=((1.5)**beta)*((1/3)*(beta**2)+beta-(23/6))-((.5)**beta)*((4/3)*(beta**2)+4*beta+(2/3))
    B_L_Min[N-3,0]=((.5)**beta)*((1/3)*(beta**2)+beta+(1/6))
    return aslinearoperator(B_L_Min)

def K_plus_R(N,h,gamma,beta,x=0):
    k_left_non_lin=lambda x: 1+x
    K_m_ss_nonlin=k_left_non_lin(x)
    constant=((1-gamma)*(h**(beta-1)))/(2*math.gamma(beta+3))
    K_plus_diag=constant*diag(K_m_ss_nonlin[1:],k=0)
    return aslinearoperator(K_plus_diag)


def K_Min_R(N,h,gamma,beta,x=0):
    k_left_non_lin=lambda x: 1+x
    K_m_ss_nonlin=k_left_non_lin(x)
    constant=((1-gamma)*(h**(beta-1)))/(2*math.gamma(beta+3))
    K_min_diag=constant*diag(K_m_ss_nonlin[:N],k=0)
    return aslinearoperator(K_min_diag)


def K_plus_L(N,h,gamma,beta,x=0):
    k_left_non_lin=lambda x: 1+x**3
    K_m_ss_nonlin=k_left_non_lin(x)
    constant=(gamma*(h**(beta-1)))/(2*math.gamma(beta+3))
    K_plus_diag=constant*diag(K_m_ss_nonlin[1:],k=0)
    return aslinearoperator(K_plus_diag)


def K_Min_L(N,h,gamma,beta,x=0):
    k_left_non_lin=lambda x: 1+x**3
    K_m_ss_nonlin=k_left_non_lin(x)
    constant=((gamma)*(h**(beta-1)))/(2*math.gamma(beta+3))
    K_min_diag=constant*diag(K_m_ss_nonlin[:N],k=0)
    return aslinearoperator(K_min_diag)


