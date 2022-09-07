from re import M
from h_mass_matrix import MassMatrix
from h_force_matrix import Force_Matrix
from h_stiff_matrix import StiffMatrix
from h_mesh import Mesh
import scipy
from scipy.sparse import csc_matrix
import numpy as np

class Final_Solution():
    def __init__(self,a,b,N,t_0,t_m,M,gamma,beta,theta):
        self.mass=MassMatrix(a,b,N,t_0, t_m,M)
        self.force=Force_Matrix(a,b,N,t_0,t_m,M)
        self.stiff=StiffMatrix(a,b,N,t_0,t_m,M,gamma,beta)
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.theta=theta
        self.N=N
        self.M=M
    def u_zero(self,x,t=0):
        return np.exp(-(x-1)**2/(2*.08**2))
    def CGS(self):
        u_0=self.u_zero(self.mesh.mesh_points()[1:self.N+1])
        u=np.zeros((self.N,self.M+1))
        u[:,0]=u_0
        for (i,t) in enumerate(self.mesh.time()[1:]):
            u_init=u[:,i]
            b=np.matmul((self.mass.Construct()+(1-self.theta)*self.mesh.delta_t()*self.stiff.B(self.mesh.time()[i])),u[:,i])+self.mesh.delta_t()*self.force.Construct()
            A=csc_matrix(self.mass.Construct()-(self.theta)*self.mesh.delta_t()*self.stiff.B(t))
            x,exit_code=scipy.sparse.linalg.cgs(A,b)
            if exit_code !=0:
                print("Failed convergence")
                break
            else:
                u[:,i+1]=x
        return u