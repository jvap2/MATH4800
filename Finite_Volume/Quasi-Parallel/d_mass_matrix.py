import cupy as cp
from d_mesh import Mesh



class MassMatrix():
    def __init__(self,a,b,N,t_0,t_m,M):
        self.mesh=Mesh(a,b,N,t_0,t_m,M)
        self.N=N
    def Construct(self):
        x=self.mesh.mesh_points()
        mid=self.mesh.midpoints()
        h=self.mesh.silengths()
        middle_diag=cp.zeros(self.N)
        upper_diag=cp.zeros(self.N-1)
        lower_diag=cp.zeros(self.N-1)
        middle_diag[:]=(1/(2*h[0:self.N]))*((x[1:self.N+1]**2)-2*x[0:self.N]*(x[1:self.N+1]-mid[0:self.N])-mid[0:self.N]**2)
        middle_diag[:]+=(1/(2*h[1:self.N+1]))*((x[1:self.N+1]**2)-2*x[2:self.N+2]*(x[1:self.N+1]-mid[1:self.N+1])-mid[1:self.N+1]**2)
        lower_diag[:]=(1/(2*h[1:self.N]))*(x[2:self.N+1]**2-2*x[2:self.N+1]*mid[1:self.N]+mid[1:self.N]**2)
        upper_diag[:]=(1/(2*h[1:self.N]))*(mid[1:self.N]**2-2*x[1:self.N]*mid[1:self.N]+x[1:self.N]**2)
        M=cp.diag(middle_diag, k=0)+cp.diag(upper_diag,k=1)+cp.diag(lower_diag,k=-1)
        return M
    def Construct_Prob_1(self):
        M=cp.zeros((self.N))
        M[(self.N)//2]=1
        M=cp.diag(M,k=0)
        return M
    def Construct_Lump(self):
        x=self.mesh.mesh_points()
        mid=self.mesh.midpoints()
        h=self.mesh.silengths()
        middle_diag=cp.zeros(self.N)
        middle_diag[:]=(1/(2*h[0:self.N]))*(x[1:self.N+1]**2-2*x[0:self.N]*(x[1:self.N+1]-mid[0:self.N])-mid[0:self.N]**2)
        middle_diag[:]+=(1/(2*h[1:self.N+1]))*(x[1:self.N+1]**2-2*x[2:self.N+2]*(x[1:self.N+1]-mid[1:self.N+1])-mid[1:self.N+1]**2)
        middle_diag[1:]+=(1/(2*h[1:self.N]))*(x[2:self.N+1]**2-2*x[2:self.N+1]*mid[1:self.N]+mid[1:self.N]**2)
        middle_diag[:self.N-1]+=(1/(2*h[1:self.N]))*(mid[1:self.N]**2-2*x[1:self.N]*mid[1:self.N]+x[1:self.N]**2)
        M=cp.diag(middle_diag, k=0)
        return M
    def Construct_Cubic(self):
        h=self.mesh.silengths()[0]
        middle_diag=cp.ones(self.N)*.807292*h
        one_off_diag=cp.ones(self.N-1)*.1145835*h
        two_off_diag=cp.ones(self.N-2)*(-.0182292*h)
        M=cp.sparse.diags(middle_diag,offsets=0)+cp.sparse.diags(one_off_diag,offsets=1)+cp.sparse.diags(one_off_diag,offsets=-1)+\
            cp.sparse.diags(two_off_diag,offsets=2)+cp.sparse.diags(two_off_diag,offsets=-2)
        return M

mass=MassMatrix(0,1,5,0,1,3)
print(mass.Construct_Cubic())
