from cmath import inf
import imp
from typing import Final
from c_view import First_View
import tkinter as tk
from c_mesh import Mesh
from c_mass_matrix import MassMatrix
from c_stiff_matrix import StiffMatrix
from c_solve import Final_Solution
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cupy as cp
import matplotlib.cm as cm
import time



class Controller():
    def __init__(self):
        self.root=tk.Tk()
        self.root.title("Fractional Diff Eq")
        self.root.geometry("500x600")
        self.view=First_View(self.root)
        self.view.mesh_button.bind("<Button>", self.make_mesh)
    def run(self):
        self.root.mainloop()
    def make_mesh(self, event):
        print("In progress...")
        a=float(self.view.A.get())
        b=float(self.view.B.get())
        N=int(self.view.N.get())
        t_0=float(self.view.tzero.get())
        t_m=float(self.view.tM.get())
        M=int(self.view.M.get())
        gamma=float(self.view.gamma.get())
        beta=float(self.view.beta.get())
        theta=float(self.view.theta.get())
        mesh=Mesh(a,b,N,t_0,t_m,M)
        sol=Final_Solution(a,b,N,t_0,t_m,M,gamma,beta,theta)
        u=np.zeros((mesh.NumofSubIntervals()+2,M+1))
        u_true=np.zeros((mesh.NumofSubIntervals()+2))
        start=time.time()
        u[1:mesh.NumofSubIntervals()+1,:]=sol.Parareal_1()
        end=time.time()
        time_inv=end-start
        x_np=cp.asnumpy(mesh.mesh_points())
        x,t=np.meshgrid(x_np,cp.asnumpy(mesh.time_points()))
        fig=plt.figure(1)
        ax=plt.axes(projection='3d')
        ax.plot_surface(x,t,np.transpose(u),cmap="plasma")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(f'FDE with dirac \u03b4 with anomolous diffusion\n \u03b2={beta},\u03b3={gamma},\u03b8={theta},N={N},M={M}\nUsing Matrix Inverse')
        plt.show()
    




        