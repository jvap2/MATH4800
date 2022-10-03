import imp
from typing import Final
from h_num_view import First_View
import tkinter as tk
from h_mesh import Mesh
from h_mass_matrix import MassMatrix
from h_stiff_matrix import StiffMatrix
from h_solve import Final_Solution
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from h_norms import Norm
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
        start=time.time()
        u[1:mesh.NumofSubIntervals()+1,:]=sol.Parareal_1()
        stop=time.time()
        time_cgs=stop-start
        u_true=np.zeros((mesh.NumofSubIntervals()+2))
        u_true=sol.True_Sol()
        x_np=mesh.mesh_points()
        x,t=np.meshgrid(mesh.mesh_points(),mesh.time())
        print(np.shape(x),np.shape(t))
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot_surface(x,t,np.transpose(u),cmap="plasma")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        plt.show()
        fig,ax=plt.subplots(1,2, figsize=(8,8))
        ax[0].plot(x_np,u_true)
        ax[1].plot(x_np,u[:,-1])
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('Behavior of True Solution, t=1')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_title('Behavior of Approximate Solution, t=1')
        plt.show()
        norm_2_ss,norm_inf_ss=Norm(x_np,u[:,-1])[0],Norm(x_np,u[:,-1])[1]
        print(f"\u0394t={mesh.delta_t()},h={x_np[1]-x_np[0]}")
        print("Norms with CGS")
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"Steady State L\u2082: {norm_2_ss}\nSteady State L\u221e:{norm_inf_ss}")
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"CGS Computational time with \u03b2={beta},\u03b3={gamma},\u03b8={theta},\u0394t={mesh.delta_t()},h={x_np[1]-x_np[0]}:\n{time_cgs} seconds")
