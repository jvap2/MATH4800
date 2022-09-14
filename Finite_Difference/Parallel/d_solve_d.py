import d_mesh_d,d_nsprmat_d,d_sink_d
import cupy as cp
import cupyx


class FD_Solve():
    def __init__(self,a,b,N,t_0,t_m,M,alpha):
        self.mesh=d_mesh_d.Mesh(a,b,N,t_0,t_m,M)
        self.sink=d_sink_d.Sink_Matrix(a,b,N,t_0,t_m)
        self.B=d_nsprmat_d.B_mat(a,b,N,t_0,t_m,M,alpha)
        self.N=N
        self.M=M
    def u_init(self):
        u_0=cp.zeros(self.N+1)
        u_0[(self.N)//2]=1
    def dirac_B_mat(self):
        B=cp.zeros(shape=(self.N+1,self.N+1))
        B[self.N/2,self.N/2]=1
    def sol(self):
        u=cp.zeros((self.N+1,self.M+1))
        u[:,0]=self.u_init()
        for (i,t) in enumerate(self.mesh.time()[0:self.M]):
            if i==0:
                u[:,i+1]=cp.matmul(self.dirac_B_mat(),u[:,0])+self.mesh.delta_t()*self.sink.Construct()
            else:
                u[:,i+1]=cp.matmul(self.B.Construct(),u[:,i])+self.mesh.delta_t()*self.sink.Construct()
        return u
