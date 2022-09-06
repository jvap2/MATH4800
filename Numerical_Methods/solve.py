from re import M
from mass_matrix import MassMatrix
from force_matrix import Force_Matrix
from stiff_matrix import StiffMatrix
from mesh import Mesh
import scipy
import numpy as np

class Final_Solution(MassMatrix,Force_Matrix,StiffMatrix):
    def __init__(self,a,b,N,t_0,t_m,M,gamma,beta):
        super().__init__(a,b,N,t_0,t_m,M,gamma,beta)
    def CGS():
        u=lambda x,t: np.exp((x-1)**2/(2*.08**2))
        u_0=u(MassMatrix.mesh_points()[1:MassMatrix.NumofSubIntervals()+1],MassMatrix.time()[0])
        
