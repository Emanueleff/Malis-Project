from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

class RidgeRegression:

    """
    Ridge Regression model implementation.

    Parameters:
    - l: Regularization parameter (default = 0)

    Attributes:
    - X: Training input data
    - y: Training output data
    - weights: Model weights
    - l: Regularization parameter

    Methods:
    - train(X, y): Train the Ridge Regression model
    - predict(X_test): Make predictions using the trained model
    - MSE(y_pred, y_test): Calculate the Mean Squared Error between predicted and actual values
    """

    def __init__(self, l=0):
        """
        Initialize the Ridge Regression model.

        Parameters:
        - l: Regularization parameter (default = 0)
        """
        self.X = []
        self.y = []
        self.weights = []
        self.l = l

    def train(self, X, y):
        """
        Train the Ridge Regression model.

        Parameters:
        - X: Training input data
        - y: Training output data
        """
        #self.X = PolynomialFeatures(1).fit_transform(X)
        self.X = X
        self.y = y
        I = np.identity(self.X.shape[1])

        # Do not regularize the bias term
        I[0][0] = 0
        self.weights = np.linalg.inv((self.X.T) @ self.X + self.l * I) @ (self.X.T) @ self.y

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        - X_test: Test input data

        Returns:
        - y_pred: Predicted output values
        """
        return np.dot(X_test, self.weights)

    def MSE(self, y_pred, y_test):
        """
        Calculate the Mean Squared Error between predicted and actual values.

        Parameters:
        - y_pred: Predicted output values
        - y_test: Actual output values

        Returns:
        - mse: Mean Squared Error
        """
        return np.mean((y_test - y_pred) ** 2)
        
    
  