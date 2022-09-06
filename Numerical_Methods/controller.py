import imp
from typing import Final
from num_view import First_View
import tkinter as tk
from mesh import Mesh
from mass_matrix import MassMatrix
from stiff_matrix import StiffMatrix
from solve import Final_Solution


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
        try:
            a=float(self.view.A.get())
            b=float(self.view.B.get())
            N=int(self.view.N.get())
            t_0=float(self.view.tzero.get())
            t_m=float(self.view.tM.get())
            M=int(self.view.M.get())
            gamma=float(self.view.gamma.get())
            beta=float(self.view.beta.get())
            omega=float(self.view.omega.get())
            sol=Final_Solution(a,b,N,t_0,t_m,M,gamma,beta,omega)
        except:
            print("Improper inputs")

        