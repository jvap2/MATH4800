from cmath import inf
import imp
from typing import Final
from d_num_view import First_View
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
from d_norms import Left_True_Solution



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
        u_cgs=np.zeros((mesh.NumofSubIntervals()+2,M+1))
        # u_true=np.zeros((mesh.NumofSubIntervals()+2))
        # start=time.time()
        # u[1:mesh.NumofSubIntervals()+1,:]=sol.MatInv()
        # end=time.time()
        # time_inv=end-start
        # u_cgs[1:mesh.NumofSubIntervals()+1,:]=sol.Solve_Cubic_1()
        u_cub=np.zeros(shape=(mesh.NumofSubIntervals()+2,1))
        u_lin=np.zeros(shape=(mesh.NumofSubIntervals()+2,1))
        u_cub[1:mesh.NumofSubIntervals()+1,0]=sol.Steady_State_Cubic_Test()
        u_lin[1:mesh.NumofSubIntervals()+1,0]=sol.Steady_State_Linear()
        x_np=cp.asnumpy(mesh.mesh_points())
        u_true=Left_True_Solution(x_np)
        x,t=np.meshgrid(x_np,cp.asnumpy(mesh.time()))
        # fig=plt.figure(1)
        # ax=plt.axes(projection='3d')
        # ax.plot_surface(x,t,np.transpose(u),cmap="plasma")
        # ax.set_xlabel('x')
        # ax.set_ylabel('t')
        # ax.set_title(f'FDE with dirac \u03b4 with anomolous diffusion\n \u03b2={beta},\u03b3={gamma},\u03b8={theta},N={N},M={M}\nUsing Matrix Inverse')
        # plt.show()
        # fig_1=plt.figure(1)
        # ax_1=plt.axes(projection='3d')
        # ax_1.plot_surface(x,t,np.transpose(u_cgs),cmap="plasma")
        # ax_1.set_xlabel('x')
        # ax_1.set_ylabel('t')
        # ax_1.set_title(f'FDE with dirac \u03b4 with anomolous diffusion\n \u03b2={beta},\u03b3={gamma},\u03b8={theta},N={N},M={M}\nUsing CGS')
        # plt.show()
        # fig,ax=plt.subplots(1,2, figsize=(8,8))
        # ax[0].plot(x_np,u_true)
        # ax[1].plot(x_np,u[:,-1])
        # ax[0].set_xlabel('x')
        # ax[0].set_ylabel('y')
        # ax[0].set_title('Behavior of True Solution, t=1')
        # ax[1].set_xlabel('x')
        # ax[1].set_ylabel('y')
        # ax[1].set_title('Behavior of Approximate Solution, t=1')
        # plt.show()
        fig,ax=plt.subplots(1,2, figsize=(8,8))
        ax[0].plot(x_np,u_true)
        # ax[1].plot(x_np,u_cgs[:,-1])
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
        # ax[1].plot(x_np,u_cgs[:,-1])
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
        print(f"\u0394t={mesh.delta_t()},h={x_np[1]-x_np[0]}")
        print("Norms with Mat Inverse, Cubic ")
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"Steady State L\u221e:{norm_inf_ss}")
        print("-----------------------------------------------------------------------------------------------------------")
        print("Norms with Mat Inverse, Linear")
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"Steady State L\u221e:{norm_inf_ss_lin}")
        # print("-----------------------------------------------------------------------------------------------------------")
        # print(f"MatInv Computational time with \u03b2={beta},\u03b3={gamma},\u03b8={theta},\u0394t={mesh.delta_t()},h={x_np[1]-x_np[0]}:\n{time_inv} seconds")
        print("-----------------------------------------------------------------------------------------------------------")
        # print(f"CGS Computational time with \u03b2={beta},\u03b3={gamma},\u03b8={theta},\u0394t={mesh.delta_t()},h={x_np[1]-x_np[0]}:\n{time_cgs} seconds")






        