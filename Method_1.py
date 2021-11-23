#------------------------------------------------
# Solution of: Newton - Raphson (method 1 and 2)
#------------------------------------------------
import numpy as np
import sympy as sym
import math

def NR(tol, U0, Ni):
    u, v, w = sym.Symbol("u"), sym.Symbol("v"), sym.Symbol("w")
    #--- vector -->  F(U)
    # f = | f[0]   f[1]    f[2]  |
    f11 = [u**2 + v**2 - 5*(sym.sin(w*np.pi/10))**2,
           u**2 + v**2 - sym.exp(w/5), u - v**2 -(1/9)*(w-10)**2]

    # --- vector -->  R
    R = [1.0, 1.0, -1.0]

    # --- Tangent Matrix ---
    #kt = | df[0]/du   df[0]/dv   df[0]/dw |
    #     | df[1]/du   df[1]/dv   df[1]/dw |
    #     | df[2]/du   df[2]/dv   df[2]/dw |
    kt11 = ([[f11[0].diff(u), f11[0].diff(v), f11[0].diff(w)],
             [f11[1].diff(u), f11[1].diff(v), f11[1].diff(w)],
             [f11[2].diff(u), f11[2].diff(v), f11[2].diff(w)]])

    # --- Solution ---
    U, error, k, c = U0, 1.0, 0, 0
    # U = | u  v  w |
    KT = np.zeros((3,3))
    du, F = np.zeros((3)), np.zeros((3))
    errk = []
    Uk = []

    while (error >= tol) and (k <100):
        k += 1
        for i in range(0,3,1):

            if (k == 1) or (Ni == 1) : #------------------ NR (1)
                KT[i,:] = [kt11[i][0].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                           kt11[i][1].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                           kt11[i][2].subs([(u, U[0]), (v, U[1]), (w, U[2])])]
            elif (k/Ni == round(k/Ni)) and (Ni>1):#---- NR Modified (2)
                KT[i, :] = [kt11[i][0].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                            kt11[i][1].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                            kt11[i][2].subs([(u, U[0]), (v, U[1]), (w, U[2])])]
                c += 1

            F[i] = f11[i].subs([(u, U[0]), (v, U[1]), (w, U[2])])

        # du = inv(KT)* (R-F)
        invKT = np.linalg.inv(KT)
        RF = R - F
        for i in range(0,3,1):
            du[i] = invKT[i,0]*RF[0] + invKT[i,1]*RF[1] + invKT[i,2]*RF[2]

        U = du + U
        errk.append( np.linalg.norm(du)/np.linalg.norm(U) )
        error = errk[k-1]
        Uk.append(U)

    return error, errk, Uk, k, U
