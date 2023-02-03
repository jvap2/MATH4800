from cmath import inf
import imp
from typing import Final
from d_num_view_SS import First_View
import tkinter as tk
from d_mesh import Mesh
from d_mass_matrix import MassMatrix
from d_stiff_matrix import StiffMatrix
from d_solve import Final_Solution
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cupy as cp
import matplotlib.cm as cm
import d_norms
import time
from d_norms import Left_True_Solution, Right_True_Solution,Left_Ex_1,Left_Ex_2



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
        gamma=float(self.view.gamma.get())
        beta=float(self.view.beta.get())
        mesh=Mesh(a,b,N)
        sol=Final_Solution(a,b,gamma,beta,N)
        u_cub=np.zeros(shape=(mesh.NumofSubIntervals()+2,1))
        u_lin=np.zeros(shape=(mesh.NumofSubIntervals()+2,1))
        u_cub[1:mesh.NumofSubIntervals()+1,0]=sol.Steady_State_Cubic_Test()
        u_lin[1:mesh.NumofSubIntervals()+1,0]=sol.Steady_State_Linear()
        x_np=cp.asnumpy(mesh.mesh_points())
        u_true=Left_Ex_2(x_np)
        x,t=np.meshgrid(x_np,cp.asnumpy(mesh.time()))
        fig,ax=plt.subplots(1,2, figsize=(8,8))
        ax[0].plot(x_np,u_true)
        ax[1].plot(x_np,u_cub,'*')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('Behavior of True Solution')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_title('Behavior of Approximate Cubic Solution SS')
        plt.grid()
        plt.show()
        fig_2,ax_2=plt.subplots(1,2, figsize=(8,8))
        ax_2[0].plot(x_np,u_true)
        ax_2[1].plot(x_np,u_lin)
        ax_2[0].set_xlabel('x')
        ax_2[0].set_ylabel('y')
        ax_2[0].set_title('Behavior of True Solution')
        ax_2[1].set_xlabel('x')
        ax_2[1].set_ylabel('y')
        ax_2[1].set_title('Behavior of Approximate Linear Solution SS')
        plt.grid()
        plt.show()
        norm_inf_ss=d_norms.Norm_SS(x_np,u_cub)
        norm_inf_ss_lin=d_norms.Norm_SS(x_np,u_lin)
        print(f"h={x_np[1]-x_np[0]}")
        print("Norms with Mat Inverse, Cubic ")
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"Steady State L\u221e:{norm_inf_ss}")
        print("-----------------------------------------------------------------------------------------------------------")
        print("Norms with Mat Inverse, Linear")
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"Steady State L\u221e:{norm_inf_ss_lin}")
        print("-----------------------------------------------------------------------------------------------------------")





        