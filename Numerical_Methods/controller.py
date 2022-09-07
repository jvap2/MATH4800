import imp
from typing import Final
from num_view import First_View
import tkinter as tk
from mesh import Mesh
from mass_matrix import MassMatrix
from stiff_matrix import StiffMatrix
from solve import Final_Solution
import numpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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
        u=cp.zeros((mesh.NumofSubIntervals()+2,M+1))
        u[1:mesh.NumofSubIntervals()+1,:]=sol.CGS().get()
        x,t=cp.meshgrid(mesh.mesh_points(),mesh.time())
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot_surface(x,t,cp.transpose(u),cmap="plasma")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        plt.show()

        