import scipy
from mesh import Mesh
from scipy import integrate


class Force_Matrix(Mesh):
    def __init__(self,a,b,N):
        super().__init__(a,b,N)
    def force_func(self,x,t):
        pass
    def Construct(self):
        pass
