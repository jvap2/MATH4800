# MATH4800
Numerical Methods to describe anomalous, Non-Fickian, diffusion with fractional differential equations. Based on the paper "Fast Finite Volume Methods for Space-Fractional Diffusion Equations"(https://www.aimsciences.org/journals/displayArticlesnew.jsp?paperID=11120). The intent is to replicate the proposed algorithm to model anomolous diffusion, and upon the creation a successful model, a Deep Learning model will be constructed. The entire aim of this paper, and the consequential deep learning model, is to show a efficable method that is computationally efficient. This will be a 6 month-1 year long project.

If you have a machine that does not have a GPU capable of working with CUDA, i.e. a Mac, you can only work with the sequential folder

Needed Dependences:<br>
  -Sequential:<br>
    > numpy<br>
    > scipy<br>
    > joblib<br>
  -Quasi Parallel:<br>
    > cupy<br>
    > cupyx<br>
    > joblib<br>
 -CUDA:<br>
    > numba<br>
    > numpy<br>
  
  
To run the code, just type run the main file in the different folders, depedning on which folder you are using, e.g. run `python h_main.py` in the Sequentail Folder. Enter the initial values as well. There is still room to make this more user friendly, which will be done over time.<br>
## Happy Trails!

