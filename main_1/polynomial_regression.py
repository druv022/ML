import numpy as np
import matplotlib.pyplot as plt


class polynomial_regression():
    """description of class"""

    def __init__(self):
        return

    def designMatrix(self,x,m):
        """designMatrix(self,x,m)
        returns np.asarray(design matrix)

        Parameters
        -----------------------
        x: np array of input
        m: highest order of polynomial

        Returns
        -----------------------
        design matrix: np array

        Note
        -----------------------
        The feature matrix or design matrix"""

        phi = []

        for i in x:
            matric = []
            for j in range(0, m + 1):
                matric.append(np.power(i,j))
            phi.append(matric)
        return np.asarray(phi)

    def fit_polynomial(self,x,t,m,lambda_reg=0):
        """ fit_polynomial(self,x,t,m,lambda_reg)
        returns w_ml, design_matrix

        Parameters
        ---------------------
        x: np array of inputs
        t: np array of targets
        m: highest order of polynomial
        lambda_reg: regularization parameter

        Returns
        --------------------
        w_ml: np array of weights of maximum-likelihood
        design_matrix: np 2D array

        This methods finds the maximum-likelihood solution of a M-th order polynomial for
        some datasetx. The error function minimised wrt w is squared error. Bishop 3.1.1"""

        phi = self.designMatrix(x,m)
        phi_trans = np.transpose(phi)

        a = phi_trans.dot(phi) + lambda_reg*np.identity(phi.shape[1])
        b = np.linalg.inv(a)
        c = b.dot(phi_trans)

        w_ml = c.dot(t)

        return w_ml, phi


