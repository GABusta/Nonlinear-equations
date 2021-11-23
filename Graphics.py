#------------------------------------------------
#--------------    Graphics    ------------------
#------------------------------------------------
import matplotlib.pyplot as plt

def Plot_results(errk1, errk2, errk3, errk4, errk5, errk6, errk7, k):
    plt.semilogy(errk1); plt.semilogy(errk2); plt.semilogy(errk3)
    plt.semilogy(errk4); plt.semilogy(errk5); plt.semilogy(errk6)
    plt.semilogy(errk7,'--k')
    plt.xlabel("Iterations"); plt.ylabel("error")
    plt.legend(['Newton Raphson','NR - Modif. (Ni = 2)',
                'NR - Modif. (Ni = 4)','NR - Modif. (Ni = 8)',
                'NR - Modif. (Ni = 16)','NR - Modif. (Ni = 32)',
                'BGFS - QuasiNewton'], loc ="upper right")
    plt.show()
    a=2
    return a