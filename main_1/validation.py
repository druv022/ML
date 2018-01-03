import polynomial_regression
import numpy as np



class validation(object):
    """description of class"""

    def __init__(self):
        self.M = [0,1,2,3,4,5,6,7,8,9,10]
        self.lambda_reg = [np.exp(-10),np.exp(-9),np.exp(-8),np.exp(-7),np.exp(-6),np.exp(-5),np.exp(-4),np.exp(-3),np.exp(-2),np.exp(-1),np.exp(-0)]


    def predict_error(self, x_train, x_valid, t_train, t_valid, m, lambda_reg):
        """predict_error(self, x_train, x_valid, t_train, t_valid, m, lambda_reg)
        return prediction_error

        Parameters
        ------------------
        x_train: np array of training set
        x_valid: np array of validation set
        t_train: np array of training target 
        t_valid: np array of validation target

        Returns
        ------------------
        prediction_error: mean squared error

        Note:
        This methods evaluates the mean squared error of training set wrt to validation set
        """

        poly_reg = polynomial_regression.polynomial_regression()
        w_ml, phi = poly_reg.fit_polynomial(x_train,t_train, m, lambda_reg)

        phi_valid = poly_reg.designMatrix(x_valid,m)
        # reg_ = lambda_reg*np.dot(np.transpose(w_ml),w_ml)

        y = phi_valid.dot(w_ml)
        pred_err = np.sum((t_valid - y) ** 2) / (x_valid.size)

        return pred_err

    def kfold_indices(self, n,k):
        """kfold_indices(self, n,k)
        return training_folds, validation_folds

        Parameters
        ---------------------
        n: number of sample points
        k: number of fold in cross-validation

        Returns
        ---------------------
        training_folds: list of list which contains the indices of training example 
        validation_folds: list of list which contains the indices of validation set

        Note:
        This methods shuffle the whole training set indices into two sets namely training set 
        and validation set for k-folds cross-validation. 
        """
        
        all_indices = np.arange(n,dtype=int)
        np.random.shuffle(all_indices)
        idx = [int(i) for i in np.floor(np.linspace(0,n,k + 1))]
        train_folds = []
        valid_folds = []

        for fold in range(k):
            valid_indices = all_indices[idx[fold]:idx[fold + 1]]
            valid_folds.append(valid_indices)
            train_folds.append(np.setdiff1d(all_indices, valid_indices))

        return train_folds, valid_folds

    def find_best_m_and_lambda(self, x, t, n, k=10):
        """find_best_m_and_lambda(self, x, t, n, k=10)
        return m_best, lambda_best
        
        Parameters
        ---------------------
        x: np array of training set
        t: np array of target set
        n: number of training examples
        k: number of folds in cross-validation, default = 10

        Returns
        ---------------------
        m_best: Best polynomial order of m
        lambda_best: Best value of regularization parameter which minimizes the prediction error for the m_best

        Note:
        ---------------------
        This method returns the best value of polynomial order and the corresponding regularization parameter.
        The set of values of m and lambda is predefined in the constructor. 
        """

        m = self.M
        l_reg = self.lambda_reg

        train_folds, valid_folds = self.kfold_indices(n,k)
        i = 0
        j = 0
        l = 0
        temp_pred_err = np.inf

        while l < k:
            for i in m:
                for j in l_reg:
                    pred_err = self.predict_error(x[train_folds[l]],x[valid_folds[l]],t[train_folds[l]],t[valid_folds[l]],i,j)
                    if pred_err < temp_pred_err:
                        lambda_best = j
                        m_best = i
                        temp_pred_err = pred_err
            l += 1

        return m_best, lambda_best


