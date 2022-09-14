import cupy as cp
from d_mesh_d import Mesh

class Sink_Matrix():
    def __init__(self,a,b,N,t_0,t_m,M):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=N
    def Construct(self):
        force=cp.zeros(self.N+1)
        return force