import scipy
from mesh import Mesh
from scipy import integrate
import numpy as np


class Force_Matrix(Mesh):
    def __init__(self,a,b,N,t_0,t_m,M):
        super().__init__(a,b,N,t_0,t_m,M)
    def force_func(self,x,t):
        pass
    def Construct(self):
        force=np.zeros(self.N)
        return force
