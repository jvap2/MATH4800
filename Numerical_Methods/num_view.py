import tkinter as tk
from tkinter import ttk


class First_View():
    def __init__(self, Master):
        self.Frame=ttk.Frame(Master, padding=20)
        self.Frame.pack()
        self.style = ttk.Style()
        self.style.configure("BW.TLabel", foreground="black", background="white")
        self.Mesh=ttk.Frame(self.Frame, padding=10)
        self.Mesh.pack()
        self.Mesh_Label=ttk.Label(self.Mesh, text="Input domain and Number of SubIntervals", style="BW.TLabel")
        self.Mesh_Label.grid(row=0, column=0)
        self.Mesh_Label.pack()
        self.A=tk.StringVar(self.Frame)
        self.B=tk.StringVar(self.Frame)
        self.N=tk.StringVar(self.Frame)
        self.A_label=ttk.Label(self.Mesh, text="x_0", style="BW.TLabel")
        self.A_label.pack()
        self.input_A=ttk.Entry(self.Mesh,textvariable=self.A)
        self.input_A.pack()
        self.B_label=ttk.Label(self.Mesh, text="x_N", style="BW.TLabel")
        self.B_label.pack()
        self.input_B=ttk.Entry(self.Mesh,textvariable=self.B)
        self.input_B.pack()
        self.N_label=ttk.Label(self.Mesh, text="x_0", style="BW.TLabel")
        self.N_label.pack()
        self.input_N=ttk.Entry(self.Mesh,textvariable=self.N)
        self.input_N.pack()
