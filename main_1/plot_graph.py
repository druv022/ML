import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import generate
import polynomial_regression

class plot_graph():
    """description of class"""

    def __init__(self):
        return

    def plot_poly_reg(self):

        z = 2 * np.pi * np.linspace(0,1,10000)
        N = 10
        M = [0,2,4,8]
        gen = generate.generate()
        x,t = gen.generate_cosine(N)
        n = 1
        for i in M:
            plt.subplot(2,2,n)
            plt.plot(z,np.cos(z))
            plt.plot(x,t,'o')
            poly_reg = polynomial_regression.polynomial_regression()
            w, phi = poly_reg.fit_polynomial(x,t,i)
            phi1 = poly_reg.designMatrix(z,i)
            y = phi1.dot(w)
            plt.plot(z,y)
            plt.xticks([0,np.pi,2 * np.pi],[r'$0$',r'$\pi$',r'$2\pi$'])
            plt.yticks([-1,0,1])
            n+=1
            plt.text(1, 1, 'M =' + str(i) , ha='center', va='center',size=8, alpha=.5)

        plt.show()







