from re import M
from mass_matrix import MassMatrix
from force_matrix import Force_Matrix
from stiff_matrix import StiffMatrix
from mesh import Mesh
import scipy
from scipy import csc_matrix
import numpy as np

class Final_Solution():
    def __init__(self,a,b,N,t_0,t_m,M,gamma,beta,omega):
        self.mass=MassMatrix(a,b,N,t_0, t_m,M)
        self.force=Force_Matrix(a,b,N,t_0,t_m,M)
        self.stiff=StiffMatrix(a,b,N,t_0,t_m,M,gamma,beta)
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.omega=omega
        self.N=N
        self.M=M
    def u_zero(self,x,t=0):
        return np.exp((x-1)**2/(2*.08**2))
    def CGS(self):
        u_0=self.u_zero(MassMatrix.mesh_points()[1:MassMatrix.NumofSubIntervals()+1])
        u=np.zeros((self.N,self.M))
        u[:,0]=u_0
        for (i,t) in enumerate(self.mesh.time()[1:]):
            u_init=u[:,i]
            b=np.matmul((self.mass.Construct()+(1-self.omega)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            A=csc_matrix(self.mass.Construct()-(self.omega)*self.mesh.delta_t()*self.stiff.B(t))
            x,exit_code=scipy.sparse.linalg.cgs(A,b)
            if exit_code !=0:
                break
            else:
                u[i+1,:]=x
        return u

