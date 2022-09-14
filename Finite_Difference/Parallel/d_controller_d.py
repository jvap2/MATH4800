from cmath import inf
import imp
from typing import Final
from d_num_view_d import First_View
import tkinter as tk
from d_mesh_d import Mesh
from d_nsprmat_d import B_mat
from d_sink_d import Sink_Matrix
from d_solve_d import FD_Solve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cupy as cp
import matplotlib.cm as cm
import d_norms_d
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
        alpha=float(self.view.alpha.get())
        mesh=Mesh(a,b,N,t_0,t_m,M)
        sol_h=FD_Solve(a,b,N,t_0,t_m,M,alpha,gamma)
        u=np.zeros((mesh.NumofSubIntervals()+1,M+1))
        u_true=np.zeros((mesh.NumofSubIntervals()+2))
        u_true_mid=np.zeros((mesh.NumofSubIntervals()+2))
        start=time.time()
        u[:,:]=sol_h.sol()
        end=time.time()
        time_inv=end-start
        u_true,u_true_mid=sol_h.true_sol()
        x_np=cp.asnumpy(mesh.mesh_points())
        x,t=np.meshgrid(x_np,cp.asnumpy(mesh.time()))
        fig=plt.figure(1)
        ax=plt.axes(projection='3d')
        ax.plot_surface(x,t,np.transpose(u),cmap="plasma")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(f'FDE with dirac \u03b4 with anomolous diffusion\n \u03b2={beta},\u03b3={gamma},\u03b8={theta},N={N},M={M}\nUsing Finite Difference')
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
        fig_2,ax_2=plt.subplots(1,2, figsize=(8,8))
        ax_2[0].plot(x_np,u_true_mid)
        ax_2[1].plot(x_np,u[:,(np.shape(u)[1]-1)//2])
        ax_2[0].set_xlabel('x')
        ax_2[0].set_ylabel('y')
        ax_2[0].set_title('Behavior of True Solution, t=.5')
        ax_2[1].set_xlabel('x')
        ax_2[1].set_ylabel('y')
        ax_2[1].set_title('Behavior of Approximate Solution, t=.5')
        plt.show()
        norm_2_ss, norm_inf_ss, norm_2_mid, norm_inf_mid=d_norms_d.Norm(x_np,u[:,-1],u_true[:])[0],d_norms_d.Norm(x_np,u[:,-1],u_true[:])[1],d_norms_d.Norm(x_np,u[:,(np.shape(u)[1]-1)//2],u_true_mid[:])[0],d_norms_d.Norm(x_np,u[:,(np.shape(u)[1]-1)//2],u_true_mid[:])[1]
        print(f"\u0394t={mesh.delta_t()},h={x_np[0]}")
        print("Norms with Mat Inverse")
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"Steady State L\u2082: {norm_2_ss}\nSteady State L\u221e: {norm_inf_ss}")
        print(f"MidPoint L\u2082: {norm_2_mid}\nMidpoint L\u221e: {norm_inf_mid}")
        print("-----------------------------------------------------------------------------------------------------------")







        