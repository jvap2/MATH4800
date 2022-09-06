import scipy
from mesh import Mesh
from scipy import integrate
import numpy as np


class Force_Matrix():
    def __init__(self,a,b,N,t_0,t_m,M):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=N
    def force_func(self,x,t):
        pass
    def Construct(self):
        force=np.zeros(self.N)
        return force
