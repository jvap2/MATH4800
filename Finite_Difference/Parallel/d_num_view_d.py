from cgitb import text
import tkinter as tk
from tkinter import ttk


class First_View():
    def __init__(self, Master):
        self.Frame=ttk.Frame(Master, padding=40)
        self.Frame.pack()
        self.style = ttk.Style()
        self.style.configure("BW.TLabel", foreground="black", background="white")
        self.Mesh=ttk.Frame(self.Frame, padding=20, style="BW.TLabel")
        self.Mesh.pack()
        self.Mesh_Label=ttk.Label(self.Mesh, text="Input domain and Num of SubIntervals", style="BW.TLabel")
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
        self.N_label=ttk.Label(self.Mesh, text="N", style="BW.TLabel")
        self.N_label.pack()
        self.input_N=ttk.Entry(self.Mesh,textvariable=self.N)
        self.input_N.pack()
        self.Time_Label=ttk.Label(self.Mesh, text="Input time domain and Num of SubIntervals", style="BW.TLabel")
        self.Time_Label.pack()
        self.tzero=tk.StringVar(self.Frame)
        self.tM=tk.StringVar(self.Frame)
        self.M=tk.StringVar(self.Frame)
        self.gamma=tk.StringVar(self.Frame)
        self.alpha=tk.StringVar(self.Frame)
        self.tzLabel=ttk.Label(self.Mesh, text="t_0", style="BW.TLabel")
        self.tzLabel.pack()
        self.input_tz=ttk.Entry(self.Mesh, textvariable=self.tzero).pack()
        self.tmLabel=ttk.Label(self.Mesh, text="t_m", style="BW.TLabel").pack()
        self.input_tm=ttk.Entry(self.Mesh, textvariable=self.tM).pack()
        self.MLabel=ttk.Label(self.Mesh, text='M', style="BW.TLabel").pack()
        self.input_m=ttk.Entry(self.Mesh, textvariable=self.M).pack()
        self.greek_label=ttk.Label(self.Mesh, text=u"Input \u03b2 and \u03b3", style="BW.TLabel").pack()
        self.alpha_label=ttk.Label(self.Mesh, text=u"\u03b1", style='BW.TLabel').pack()
        self.input_alpha=ttk.Entry(self.Mesh, textvariable=self.alpha).pack()
        self.gamma_label=ttk.Label(self.Mesh,text=u"\u03b3",style="BW.TLabel").pack()
        self.input_gamma=ttk.Entry(self.Mesh, textvariable=self.gamma).pack()
        self.A.set("-4")
        self.B.set("4")
        self.N.set("1024")
        self.tzero.set("0")
        self.tM.set("1")
        self.M.set("128")
        self.alpha.set(".2")
        self.gamma.set(".5")
        self.mesh_button=tk.Button(self.Mesh, text="Enter")
        self.mesh_button.pack()
