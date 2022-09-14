import numpy as np
from h_mesh import Mesh

class MassMatrix():
    '''
    Class for construction of Mass Matrix
    Inherites the attributes of the Mesh class
    '''
    def __init__(self,a,b,N,t_0,t_m,M):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=self.mesh.NumofSubIntervals()
    def Construct(self):
        x=self.mesh.mesh_points()
        mid=self.mesh.midpoints()
        h=self.mesh.silengths()
        middle_diag=np.zeros(self.N)
        upper_diag=np.zeros(self.N-1)
        lower_diag=np.zeros(self.N-1)
        middle_diag[:]=(1/(2*h[0:self.N]))*(x[1:self.N+1]**2-2*x[0:self.N]*(x[1:self.N+1]-mid[0:self.N])-mid[0:self.N]**2)
        middle_diag[:]+=(1/(2*h[1:self.N+1]))*(x[1:self.N+1]**2-2*x[2:self.N+2]*(x[1:self.N+1]-mid[1:self.N+1])-mid[1:self.N+1]**2)
        lower_diag[:]=(1/(2*h[1:self.N]))*(x[2:self.N+1]**2-2*x[2:self.N+1]*mid[1:self.N]+mid[1:self.N]**2)
        upper_diag[:]=(1/(2*h[1:self.N]))*(mid[1:self.N]**2-2*x[1:self.N]*mid[1:self.N]+x[1:self.N]**2)
        M=np.diag(middle_diag, k=0)+np.diag(upper_diag,k=1)+np.diag(lower_diag,k=-1)
        return M