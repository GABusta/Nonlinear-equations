#------------------------------------------------
# Solution of: BFGS method ( quasi Newton )
#------------------------------------------------
import numpy as np
import math
import sympy as sym

def BGFS(tol, U0):
    u, v, w = sym.Symbol("u"), sym.Symbol("v"), sym.Symbol("w")
    U = U0

    #--- Vector F(U) ---
    f11 = [u**2 + v**2 - 5*(sym.sin(w*np.pi/10))**2,
           u**2 + v**2 - sym.exp(w/5), u - v**2 - (1/9)*(w-10)**2]

    # --- Vector R ---
    R = [1.0, 1.0, -1.0]

    # --- Tangent Matrix ---
    #kt = | df[0]/du   df[0]/dv   df[0]/dw |
    #     | df[1]/du   df[1]/dv   df[1]/dw |
    #     | df[2]/du   df[2]/dv   df[2]/dw |
    kt11 = ([[f11[0].diff(u), f11[0].diff(v), f11[0].diff(w)],
             [f11[1].diff(u), f11[1].diff(v), f11[1].diff(w)],
             [f11[2].diff(u), f11[2].diff(v), f11[2].diff(w)]])
    KT = np.zeros((3,3))
    for i in range(0,3,1):
        KT[i, :] = [kt11[i][0].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                    kt11[i][1].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                    kt11[i][2].subs([(u, U[0]), (v, U[1]), (w, U[2])])]
    invKT = np.linalg.inv(KT)

    I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    error, k = 1.0, 0
    errk = []
    Uk = []

    # --- Solution ---
    while (error >= tol) and (k < 100):
        k += 1
        F, RF, du, df = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        #------------------------------------------------------
        for i in range(0,3,1):
            F[i] = f11[i].subs([(u, U[0]), (v, U[1]), (w, U[2])])
        RF = R - F
        for i in range(0,3,1):
            du[i] = invKT[i,0]*RF[0] + invKT[i,1]*RF[1] + invKT[i,2]*RF[2]
        U = du + U
        for i in range(0,3,1):
            df[i] = f11[i].subs([(u, U[0]), (v, U[1]), (w, U[2])]) - F[i]

        # ------------------------------------------------------
        A, ck = A_matrix(KT, du, df, I)

        # --- invKT (correction) ---
        if ck > 1E5 :
            invKT2 = np.zeros((3, 3))
            for i in range(0,3,1):
                for j in range(0,3,1):
                    for m in range(0,3,1):
                        invKT2[i][j] = invKT2[i][j] + A[m][i]*KT[i][m]*A[m][j]
            invKT = invKT2
        # --- (new KT) ---
        for i in range(0, 3, 1):
            KT[i, :] = [kt11[i][0].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                        kt11[i][1].subs([(u, U[0]), (v, U[1]), (w, U[2])]),
                        kt11[i][2].subs([(u, U[0]), (v, U[1]), (w, U[2])])]
        # --- Error ---
        errk.append(np.linalg.norm(du) / np.linalg.norm(U))
        error = errk[k - 1]
        Uk.append(U)

    return error, errk, Uk, k, U


#------------------------------------
#---------   A  matrix   ------------
#------------------------------------
def A_matrix(KT, du, df, I):
    a22, a23 = np.zeros(3), 0.0
    vv, A = np.zeros(3), np.zeros((3, 3))
    a11 = du[0] * df[0] + du[1] * df[1] + du[2] * df[2]
    # --- ck ---
    for i in range(0, 3, 1):
        for j in range(0, 3, 1):
            a22[i] = a22[i] + KT[i, j] * du[j]  # du[i]*(KT[i,0]*du[0] + KT[i,1]*du[1] +KT[i,2]*du[2])
        a23 = a23 + du[i] * a22[i]
    ck = (a11 / a23) ** 0.5
    # --- vv ---
    for i in range(0, 3, 1):
        vv[i] = -ck * (KT[i][0] * du[0] + KT[i][1] * du[1] + KT[i][2] * du[2]) - df[i]
    # --- ww ---
    ww = du / (du[0] * df[0] + du[1] * df[1] + du[2] * df[2])
    # --- A ---
    for i in range(0, 3, 1):
        for j in range(0, 3, 1):
            A[i][j] = A[i][j] + vv[i] * ww[j]
    A = A + I
    return A, ck




