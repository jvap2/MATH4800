import d_mesh_d,d_nsprmat_d,d_sink_d
import cupy as cp
import cupyx


class FD_Solve():
    def __init__(self,a,b,N,t_0,t_m,M,alpha):
        self.mesh=d_mesh_d.Mesh(a,b,N,t_0,t_m,M)
        self.sink=d_sink_d.Sink_Matrix(a,b,N,t_0,t_m)
        self.B=d_nsprmat_d.B_mat(a,b,N,t_0,t_m,M,alpha)
    def MatInv(self):
        