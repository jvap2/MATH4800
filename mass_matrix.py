import numpy as np
from mesh import Mesh

class MassMatrix(Mesh):
    def __init__(self,a,b,N):
        super().__init__(a,b,N)
    def Construct(self):
        middle_diag=np.zeros(self.N)
        upper_diag=np.zeros(self.N-1)
        lower_diag=np.zeros(self.N-1)
        middle_diag[1:self.N-1]=(1/(2*self.silengths()[0:self.N-2]))*(self.mesh_points()[1:self.N-1]**2-2*self.mesh_points()[0:self.N-2]*(self.mesh_points()[1:self.N-1]-self.midpoints()[0:self.N-2])-self.midpoints()[0:self.N-2]**2)
        middle_diag[1:self.N-1]+=(1/(2*self.silengths()[1:self.N-1]))*(self.mesh_points()[1:self.N-1]**2+2*self.mesh_points()[2:self.N]*(self.midpoints()[1:self.N-1]-self.mesh_points()[1:self.N-1])-self.midpoints()[1:self.N-1]**2)
        return middle_diag


M=MassMatrix(0,4,5)
print(M.Construct())