import numpy as np
from mesh import Mesh

class MassMatrix(Mesh):
    '''
    Class for construction of Mass Matrix
    Inherites the attributes of the Mesh class
    '''
    def __init__(self,a,b,N):
        super().__init__(a,b,N)
    def Construct(self):
        middle_diag=np.zeros(self.N)
        upper_diag=np.zeros(self.N-1)
        lower_diag=np.zeros(self.N-1)
        middle_diag[0]=(1/(2*self.silengths()[0]))*(self.midpoints()[0]**2-2*self.mesh_points()[0]*self.midpoints()[0]+self.mesh_points()[0])
        middle_diag[self.N-1]=(1/(2*self.silengths()[-1]))*(self.midpoints()[-1]**2-2*self.mesh_points()[-2]*(self.midpoints()[-1]-self.mesh_points()[-1])-self.mesh_points()[-1]**2)
        middle_diag[1:self.N-1]=(1/(2*self.silengths()[0:self.N-2]))*(self.mesh_points()[1:self.N-1]**2-2*self.mesh_points()[0:self.N-2]*(self.mesh_points()[1:self.N-1]-self.midpoints()[0:self.N-2])-self.midpoints()[0:self.N-2]**2)
        middle_diag[1:self.N-1]+=(1/(2*self.silengths()[1:self.N-1]))*(self.mesh_points()[1:self.N-1]**2+2*self.mesh_points()[2:self.N]*(self.midpoints()[1:self.N-1]-self.mesh_points()[1:self.N-1])-self.midpoints()[1:self.N-1]**2)
        lower_diag[0:]=(1/(2*self.silengths()[1:]))*(self.mesh_points()[1:self.N]**2-2*self.mesh_points()[1:self.N]*self.midpoints()[1:]+self.midpoints()[1:]**2)
        upper_diag[0:]=(1/(2*self.silengths()[:self.N-1]))*(self.midpoints()[:self.N-1]**2-2*self.mesh_points()[:self.N-1]*self.midpoints()[:self.N-1]+self.mesh_points()[:self.N-1]**2)
        M=np.diag(middle_diag,k=0)+np.diag(upper_diag, k=1)+np.diag(lower_diag,k=-1)
        return M

m=MassMatrix(0,4,5)
print(m.Construct())