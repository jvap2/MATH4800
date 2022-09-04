import tkinter as tk

class First_View():
    def __init__(self, Master):
        self.pad_small={"padx":6, "pady":6}
        self.pad_med={"padx":12, "pady": 12}
        self.pad_frame={"padx":45, "pady":45}
        self.font_small={"font":8}
        self.font_med={"font":12}
        self.font_large={"font":20}
        self.purple_fg={"fg":"MediumPurple3"}
        self.blue_bg={"bg":"midnight blue"}
        self.widget_blue={"highlightbackgroud":"midnight blue"}
        self.gray_bg={"bg":"gray17"}
        self.Option_menu_thick={"highlightthickness": 0}
        self.Label_font={"fg": "NavajoWhite2"}
        self.Frame=tk.Frame(Master)
        self.Frame.config(**self.gray_bg)
        self.Frame.grid()
        self.Mesh=tk.Frame(self.Frame)
        self.Mesh.config(**self.purple_fg)
        self.Mesh_Label=tk.Label(self.Mesh, text="Input domain and Number of SubIntervals",
        **self.font_large, **self.pad_med, **self.gray_bg)
        self.Mesh_Label.grid(row=0)
        self.A=tk.StringVar(self.Frame)
        self.B=tk.StringVar(self.Frame)
        self.N=tk.StringVar(self.Frame)
        self.A_label=tk.Label(self.Mesh, text="x_0", **self.font_med, **self.pad_med, **self.gray_bg)
        self.input_A=tk.Entry(self.Mesh,textvariable=self.A)
        self.B_label=tk.Label(self.Mesh, text="x_N", **self.font_med, **self.pad_med, **self.gray_bg)
        self.input_B=tk.Entry(self.Mesh,textvariable=self.B)
        self.N_label=tk.Label(self.Mesh, text="x_0", **self.font_med, **self.pad_med, **self.gray_bg)
        self.input_N=tk.Entry(self.Mesh,textvariable=self.N)
