import numpy as np
import math

def norm(vector):
    L_inf=np.max(np.abs(vector))
    vector[:]=vector[:]**2
    L_2=math.sqrt(float(np.sum(vector)))
    return [L_2,L_inf]

