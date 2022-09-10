import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import quad

def Norm(x_vector,y_approx,y_true):
    int_2=np.zeros(shape=(len(x_vector)),dtype=np.float64)
    int_inf=np.zeros(shape=(len(x_vector)),dtype=np.float64)
    for i in range(len(x_vector)-1):
        interp_1=interp1d(x_vector[i:i+2].squeeze(),y_approx[i:i+2].squeeze())
        interp_2=interp1d(x_vector[i:i+2].squeeze(),y_true[i:i+2].squeeze())
        int_f_2= lambda x: (interp_1(x)-interp_2(x))**2
        int_f_inf = lambda x: abs(interp_1(x)-interp_2(x))
        int_2[i]=quad(int_f_2,x_vector[i],x_vector[i+1])[0]
        int_inf[i]=quad(int_f_inf,x_vector[i],x_vector[i+1])[0]
    return [np.max(int_2),np.max(int_inf)]
