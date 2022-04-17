# https://towardsdatascience.com/ml-from-scratch-linear-polynomial-and-regularized-regression-models-725672336076
# helped with the norm: https://www.journaldev.com/45324/norm-of-vector-python

"""Module containing classes for regularization of linear models."""
import numpy as np
from scipy.linalg import lstsq
import math

def slicer(array, axis=1, range_func=None):
    if range_func is None:
        return array.take(indices=(0, array.shape[axis]), axis=axis)
    else:
        return array.take(indices=range_func, axis=axis)
    
def norm(w, norm_coef=2):
    if norm_coef == 1:
        try:
            iter(abs(w).sum())
            return max(abs(w).sum())
        except TypeError:
            return abs(w).sum()
    if norm_coef == -1:
        try:
            iter(abs(w).sum())
            return min(abs(w).sum())
        except TypeError:
            return abs(w).sum()
    else:
        return math.pow(np.power(w, norm_coef).sum(), 1/norm_coef)
                    

def mean_squared_error(y_true, y_pred, squared=True):
    """
    Mean squared error regression loss function.
    Parameters 
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    squared : bool, default=True
        If True returns MSE, if False returns RMSE.
    Returns 
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).
    """
    # Make sure inputs are numpy arrays.
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate array of errors for each instance.
    errors = np.average((y_true - y_pred) ** 2, axis=0)

    # Calculates square root of each error if squared=False.
    if not squared:
        errors = np.sqrt(errors)

    # Return average error across all instances.
    return np.average(errors)

class lp_regularization:
    """
    Add l_{p} regularization penalty to linear models.

    Regularization term:

        alpha * L_{p}

    Where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.

    Parameters
    ----------
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.
    norm_coef: int, default=2
        The functional form of the regularization

    Notes
    -----
    The bias term is not regularized and therefore should be omitted from the
    feature weights as input.  
    """
    def __init__(self, alpha=1.0, norm_coef=2):
        self.alpha = alpha
        self.norm_coef = norm_coef

    def __call__(self, w):
        """Calculate regularization term."""
        return self.alpha * norm(w, norm_coef=self.norm_coef)
    
    def grad(self, w):
        """
        Calculate gradient descent regularization term.

            alpha * w

        where alpha is the factor determining the amount of regularization and
        w is the vector of feature weights.  
        """
        gradient_penalty = np.asarray(self.alpha) * w
        # Insert 0 for bias term.
        return np.insert(gradient_penalty, 0, 0, axis=0)

class lplq_regularization:
    """
    Add (beta*lp + 1/beta*lq) regularization penalty to linear models.

    Regularization term:

        alpha * (beta * L_{p} + 1/beta * L_{q})

    Where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.

    Parameters
    ----------
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.
    norm_coef: int, default=2
        The functional form of the regularization

    Notes
    -----
    The bias term is not regularized and therefore should be omitted from the
    feature weights as input.  
    """
    def __init__(self, alpha=1.0, beta=0.5, norm_coef_one=2, norm_coef_two=1):
        self.alpha = alpha
        self.beta = beta
        self.norm_coef_one = norm_coef_one
        self.norm_coef_two = norm_coef_two

    def __call__(self, w):
        """Calculate regularization term."""
        return self.alpha * (
            self.beta * norm(w, norm_coef=self.norm_coef_one) +
            (1/self.beta) * norm(w, norm_coef=self.norm_coef_two)
        )
    
    def grad(self, w):
        """
        Calculate gradient descent regularization term.

            alpha * w

        where alpha is the factor determining the amount of regularization and
        w is the vector of feature weights.  
        """
        gradient_penalty = np.asarray(self.alpha) * w
        # Insert 0 for bias term.
        return np.insert(gradient_penalty, 0, 0, axis=0)

class Regression:
    """
    Class representing our base regression model.  
    
    Models relationship between a dependant scaler variable y and independent
    variables X by optimizing a cost function with batch gradient descent.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.

    Attributes 
    ----------
    coef_ : array of shape (n_features, 1)
        Estimated coefficients for the regression problem.
    """
    def __init__(self, n_iter=1000, lr=1e-1, threshold=1e5):
        self.n_iter = n_iter 
        self.lr = lr
        self.threshold = threshold

    def fit(self, X, y):
        """
        Fit linear model with batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples, 1)
            Target values. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # Make sure inputs are numpy arrays.
        X = np.array(X)
        y = np.array(y)
        # Add x_0 = 1 to each instance for the bias term.
        X = np.c_[np.ones((X.shape[0], 1)), X]
        # Store number of samples and features in variables.
        n_samples, n_features = np.shape(X)
        self.training_errors = []
        # Initialize weights randomly from normal distribution.
        self.coef_ = np.random.randn(n_features, 1)
        # Batch gradient descent for number iterations = n_iter.
        for _ in range(self.n_iter):
            y_preds = X.dot(self.coef_)
            # Penalty term if regularized (don't include bias term).
            regularization = self.regularization(self.coef_[1:])
            # Calculate mse + penalty term if regularized.
            cost_function = mean_squared_error(y, y_preds) + regularization
            if cost_function > self.threshold:
                break
            self.training_errors.append(cost_function) 
            # Regularization term of gradients (don't include bias term).
            gradient_reg = self.regularization.grad(self.coef_[1:])
            # Gradients of loss function.
            gradients = (2/n_samples) * X.T.dot(y_preds - y)
            gradients += gradient_reg
            # Update the weights.
            self.coef_ -= (self.lr * gradients.sum(axis=1)).reshape(-1, 1)

        return self 

    def predict(self, X):
        """
        Estimate target values using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Instances.

        Returns
        -------
        C : array of shape (n_samples, 1)
            Estimated targets per instance.
        """
        # Make sure inputs are numpy arrays.
        X = np.array(X)
        # Add x_0 = 1 to each instance for the bias term.
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return X.dot(self.coef_)


class LinearRegression(Regression):
    """
    Class representing a linear regression model.

    Models relationship between target variable and attributes by computing 
    line that minimizes mean squared error.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.     
    solver : {'bgd', 'lstsq'}, default="bgd"
        Optimization method used to minimize mean squared error in training.

        'bgd' : 
            Batch gradient descent.

        'lstsq' : 
            Ordinary lease squares method using scipy.linalg.lstsq.

    Attributes 
    ----------
    coef_ : array of shape (n_features, 1)
        Estimated coefficients for the regression problem.

    Notes
    -----
    This class is capable of being trained using ordinary least squares method
    or batch gradient descent.  See solver parameter above.
    """
    def __init__(self, n_iter=1000, lr=1e-1, solver='bgd'):
        self.solver = solver 
        # No regularization.
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iter=n_iter, lr=lr)

    def fit(self, X, y):
        """
        Fit linear regression model.

        If solver='bgd', model is trained using batch gradient descent. 
        If solver='lstsq' model is trained using ordinary least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples, 1)
            Target values. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # If solver is 'lstsq' use ordinary least squares optimization method.
        if self.solver == 'lstsq':
            # Make sure inputs are numpy arrays.
            X = np.array(X)
            y = np.array(y)
            # Add x_0 = 1 to each instance for the bias term.
            X = np.c_[np.ones((X.shape[0], 1)), X]
            # Scipy implementation of least squares.
            self.coef_, residues, rank, singular = lstsq(X, y)

            return self

        elif self.solver == 'bgd': 
            super(LinearRegression, self).fit(X, y)


class LpRegression(Regression):
    """
    Class representing a linear regression model with l2 regularization.

    Minimizes the cost fuction:

        J(w) = MSE(w) + alpha * L_{p}

    where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.
    
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.

    Attributes 
    ----------
    coef_ : array of shape (n_features, 1)
        Estimated coefficients for the regression problem.

    Notes
    -----
    This class is capable of being trained using batch gradient descent at
    current version.
    """
    def __init__(self, n_iter=100, lr=1e-2, alpha=1.0, norm_coef=2):
        self.alpha = alpha
        self.norm_coef = norm_coef
        self.regularization = lp_regularization(alpha=self.alpha, norm_coef=self.norm_coef)
        super(LpRegression, self).__init__(n_iter=n_iter, lr=lr)

class LpLqRegression(Regression):
    """
    Class representing a linear regression model with l2 regularization.

    Minimizes the cost fuction:

        J(w) = MSE(w) + alpha * (beta * L_{p} + 1/beta * L_{q})

    where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.
    
    0 < beta < 1

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.
    
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.

    Attributes 
    ----------
    coef_ : array of shape (n_features, 1)
        Estimated coefficients for the regression problem.

    Notes
    -----
    This class is capable of being trained using batch gradient descent at
    current version.
    """
    def __init__(self, n_iter=100, lr=1e-2, alpha=1.0, beta=0.5, norm_coef_one=2, norm_coef_two=1):
        self.alpha = alpha
        self.beta = beta
        self.norm_coef_one = norm_coef_one
        self.norm_coef_two = norm_coef_two
        self.regularization = lplq_regularization(
            alpha=self.alpha, beta=self.beta,
            norm_coef_one=self.norm_coef_one,
            norm_coef_two=self.norm_coef_two
        )
        super(LpLqRegression, self).__init__(n_iter=n_iter, lr=lr)

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # default is L2 norm
    elasticnet_reg = LpLqRegression(norm_coef_one=2, norm_coef_two=1)
    df = pd.DataFrame()
    df[0] = np.random.normal(50, 12, size=1000)
    df[1] = np.random.normal(60, 14, size=1000)
    df[2] = df[1] + np.random.random(size=df.shape[0])
    df["y"] = 3*df[0] + 2*df[1] + 4*df[2]
    df.head()
    X = df[[0, 1, 2]].values
    y = df["y"].values
    elasticnet_reg.fit(X, y)
    print(elasticnet_reg.predict(X))
