import scipy
from d_mesh import Mesh
from scipy import integrate
import cupy as cp


class Force_Matrix():
    def __init__(self,a,b,N,t_0,t_m,M):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=N
    def Construct(self):
        force=cp.zeros(self.N)
        return force
