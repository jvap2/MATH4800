import numpy as np


class Force_Matrix():
    def __init__(self,N):
        self.N=N
    def Construct(self):
        force=np.zeros(self.N)
        return force