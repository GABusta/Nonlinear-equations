#-----------------------------------------------
#------- GABusta program - Nonlinear Eqs.-------
#-----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from Method_1 import NR
from Method_2 import BGFS
from Graphics import Plot_results
#-------------------- DATA ---------------------
tol = 1.0E-10
U0 = [1.0,1.0,7.0]

# ( 1 ) Newton - Raphson method ----------------
Ni = 1                 # Number of constans for KT
error1, errk1, Uk1, k1, U1 = NR(tol,U0, Ni)

# ( 2 ) Modified Newton - Raphson method -------
Ni = [2, 4, 8, 16, 32] # Number of constans for KT
error2, errk2, Uk2, k2, U2 = NR(tol,U0, Ni[0])
error3, errk3, Uk3, k3, U3 = NR(tol,U0, Ni[1])
error4, errk4, Uk4, k4, U4 = NR(tol,U0, Ni[2])
error5, errk5, Uk5, k5, U5 = NR(tol,U0, Ni[3])
error6, errk6, Uk6, k6, U6 = NR(tol,U0, Ni[4])

# ( 3 ) BFGS - QuasiNewton method --------------
error7, errk7, Uk7, k7, U7 = BGFS(tol,U0)

# ------------ Print Results ------------------
k = [k1, k2, k3, k4, k5, k6, k7]
a = Plot_results(errk1, errk2, errk3, errk4, errk5, errk6, errk7, k)

#print(errk)
