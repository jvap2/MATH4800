import numpy as np


class Mesh():
    def __init__(self,a,b,N):
        '''
        Mesh class will return our sandbox of data w.r.t x
        @param a: lower boundary
        @param b: upper boundary
        @param N: number of partitions
        @return NumofSubIntervals: N
        @return mesh_points: These are x_j, j=0,1,...N
        @return silengths: numpy array of the distances between mesh points
        @return mid_points: These are x_{j-1/2}, j=1,2,...N
        @return cvlenghts: Distances between end points and midpoints, in addition to distance between mid points, np.array
        '''
        self.a=a
        self.b=b
        self.N=N
    def NumofSubIntervals(self):
        return self.N
    def mesh_points(self):
        return np.linspace(self.a, self.b, self.N+1)
    def silengths(self):
        silengths=self.mesh_points()[1:]-self.mesh_points()[:self.N]
        return silengths
    def midpoints(self):
        return (self.mesh_points()[:self.N]+self.mesh_points()[1:])/2
    def cvlengths(self):
        cvlengths=np.zeros(self.N+1)
        cvlengths[0]=self.midpoints()[0]-self.a
        cvlengths[self.N]=self.b-self.midpoints()[self.N-1]
        cvlengths[1:self.N]=self.midpoints()[1:]-self.midpoints()[:self.N-1]
        return cvlengths


domain=Mesh(0,4,5)
print(f"Number of subintervals: {domain.NumofSubIntervals()}\nmesh_points:{domain.mesh_points()}\nsilengths:{len(domain.silengths())}\nmidpoints:{len(domain.midpoints())}\ncvlengths:{domain.cvlengths()}")