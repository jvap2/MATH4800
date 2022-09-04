import imp
from num_view import First_View
import tkinter as tk


class Controller():
    def __init__(self):
        self.root=tk.Tk()
        self.root.title("Fractional Diff Eq")
        self.root.resizable(0,0)
        self.root.geometry("200x200")
        self.view=First_View(self.root)
    def run(self):
        self.root.config(**self.view.gray_bg)
        self.root.mainloop()