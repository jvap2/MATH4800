import scipy
from d_mesh import Mesh
from scipy import integrate
import cupy as cp
from math import gamma
import numpy as np
from cupyx.scipy.sparse.linalg import aslinearoperator


class Force_Matrix():
    def __init__(self,a,b,N,t_0=0,t_m=0,M=0):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=N
        self.mid=cp.asnumpy(self.mesh.midpoints())
        self.points=self.mesh.mesh_points()
    def Construct(self):
        beta=.7
        f=np.zeros(self.N)
        force= lambda x: (-1/gamma(beta+3))*((2*x**beta)*((beta**2)*(x+1)+beta*(-3*x**2+x+3)-9*x**2-2*x+2))
        for i in range(self.N):
            f[i],_=integrate.quad(force,self.mid[i],self.mid[i+1])
        f=cp.array(f)
        return f
    def Construct_Right(self):
        beta=.7
        f=np.zeros(self.N)
        force_r=lambda x: -(((1-x)**(beta-1))/gamma(beta+3))*((beta**3)*(x+1)+(beta**2)*(4*(x**2)+4*x-2)+beta*(6*(x**3)+10*(x**2)-9*x-3)+2*(9*(x**3)-7*(x**2)-4*x+2))
        for i in range(self.N):
            f[i],_=integrate.quad(force_r,self.mid[i],self.mid[i+1])
        f=cp.array(f)
        return f
    def Construct_Right_LinOp(self):
        beta=.7
        f=np.zeros(self.N)
        force_r=lambda x: -(((1-x)**(beta-1))/gamma(beta+3))*((beta**3)*(x+1)+(beta**2)*(4*(x**2)+4*x-2)+beta*(6*(x**3)+10*(x**2)-9*x-3)+2*(9*(x**3)-7*(x**2)-4*x+2))
        for i in range(self.N):
            f[i],_=integrate.quad(force_r,self.mid[i],self.mid[i+1])
        f=cp.array(f)
        f=aslinearoperator(f)
        return f
    def Construct_Time(self):
        f=cp.zeros(self.N)
        return f