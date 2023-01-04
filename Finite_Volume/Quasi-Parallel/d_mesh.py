import cupy as cp


class Mesh():
    def __init__(self,a,b,N,t_0=0,t_m=0,M=0):
        '''
        Mesh class will return our sandbox of data w.r.t x
        @param a: lower boundary
        @param b: upper boundary
        @param N: number of partitions
        @return NumofSubIntervals: N
        @return mesh_points: These are x_j, j=0,1,...N
        @return silengths: cupy array of the distances between mesh points
        @return mid_points: These are x_{j-1/2}, j=1,2,...N
        @return cvlenghts: Distances between end points and midpoints, in addition to distance between mid points, cp.array
        '''
        self.a=a
        self.b=b
        self.N=N
        self.t_0=t_0
        self.t_m=t_m
        self.m=M
    def NumofSubIntervals(self):
        return self.N
    def mesh_points(self):
        return cp.linspace(self.a, self.b, self.N+2)
    def silengths(self):
        silengths=self.mesh_points()[1:]-self.mesh_points()[:self.N+1]
        return silengths
    def midpoints(self):
        return (self.mesh_points()[:self.N+1]+self.mesh_points()[1:])/2
    def time(self):
        return cp.linspace(self.t_0, self.t_m,self.m+1)
    def delta_t(self):
        return (self.time()[1]-self.time()[0])


m=Mesh(0,1,5,0,1,5)
print(cp.shape(m.midpoints()))