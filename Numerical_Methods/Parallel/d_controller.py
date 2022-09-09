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
        u_true=np.zeros((mesh.NumofSubIntervals()+2,1))
        u_true_mid=np.zeros((mesh.NumofSubIntervals()+2,1))
        u[1:mesh.NumofSubIntervals()+1,:],u_true, u_true_mid=sol.MatInv()
        x,t=np.meshgrid(cp.asnumpy(mesh.mesh_points()),cp.asnumpy(mesh.time()))
        fig=plt.figure(1)
        ax=plt.axes(projection='3d')
        ax.plot_surface(x,t,np.transpose(u),cmap="plasma")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(f'FDE with dirac \u03b4 with anomolous diffusion\n \u03b2={beta},\u03b3={gamma},\u03b8={theta},N={N},M={M}')
        plt.show()
        fig_2=plt.figure(2)
        ax_1=plt.axes()
        line, =ax_1.plot(cp.asnumpy(mesh.mesh_points()),u_true)
        line.set_label("True")
        line_1, =ax_1.plot(cp.asnumpy(mesh.mesh_points()),u[:,-1])
        line_1.set_label("Approximate")
        ax_1.legend()
        ax_1.set_xlabel('x')
        ax_1.set_ylabel('y')
        ax_1.set_title('Steady State Behavior of Approximate and True Solution')
        plt.show()
        ax_2=plt.axes()
        line_2, =ax_2.plot(cp.asnumpy(mesh.mesh_points()),u_true_mid)
        line_2.set_label("True")
        line_3, =ax_2.plot(cp.asnumpy(mesh.mesh_points()),u[:,(np.shape(u)[1]-1)//2])
        line_3.set_label("Approximate")
        ax_2.legend()
        ax_2.set_xlabel('x')
        ax_2.set_ylabel('y')
        ax_2.set_title('Steady State Behavior of Approximate and True Solution, t=.5')
        plt.show()
        print("uh(.,1):\n",u[:,-1])
        print("u(.,1):\n",u_true)
        dif_ss=u_true[:]-u[:,-1]
        print("Difference:\n",np.max(dif_ss))
        dif_mid=u_true_mid[:]-u[:,(np.shape(u)[1]-1)//2]
        norm_2_ss, norm_inf_ss, norm_2_mid, norm_inf_mid=d_norms.norm(dif_ss)[0],d_norms.norm(dif_ss)[1],d_norms.norm(dif_mid)[0],d_norms.norm(dif_mid)[1]
        print(f"Steady State L\u2082: {norm_2_ss}\nSteady State L\u221e: {norm_inf_ss}")
        print(f"MidPoint L\u2082: {norm_2_mid}\nMidpoint L\u221e: {norm_inf_mid}")




        