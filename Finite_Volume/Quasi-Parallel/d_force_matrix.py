import scipy
from d_mesh import Mesh
from scipy import integrate
import cupy as cp
from math import gamma


class Force_Matrix():
    def __init__(self,a,b,N,t_0,t_m,M):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=N
    def Construct(self):
        # force=cp.zeros(self.N)
        force= lambda x: (2*x**(.7))/(.7*gamma(.7))-(x**(-.3)/gamma(.7))
        f=force(self.mesh.midpoints()[1:])*self.mesh.silengths()[0]
        return f
