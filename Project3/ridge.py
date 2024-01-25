"""
DISCLAIMER: A first draft of the documentation of the code has been provided by ChatGPT, then checked and modified by the authors.
"""


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
        Initialize the Ridge Regression model with empty list for training input and output data and empty list for model weights

        Parameters:
        - l: Regularization parameter (default = 0)
        """
        self.X = []
        self.y = []
        self.weights = []
        self.l = l

    def embed_bias(self, X):
        """
        Adds a column of 1s to the front of the input matrix.

        """
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))
        return X

    def train(self, X, y):
        """
        Train the Ridge Regression model.

        Parameters:
        - X: Training input data with dimension (NxD), then adapted to (Nx(D+1)) in order to embed the bias term
        - y: Training output data with dimension (Nx1)
        """
        if(X.shape[0] == y.shape[0]):
            #Adapt the dataset in order to embed the bias term
            self.X = self.embed_bias(X)
            self.y = y

            #Identity matrix for Ridge Regression formula
            I = np.identity(self.X.shape[1])
            #Do not regularize the bias term
            I[0][0] = 0

            #Compute the weigths of dimension ((D+1)x1)
            self.weights = np.linalg.inv((self.X.T) @ self.X + self.l * I) @ (self.X.T) @ self.y
        else:
            print("The number of rows of X and y must be the same")

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        - X_test: Test input data

        Returns:
        - y_pred: Predicted output values
        """
        #Before making the prediction the test data are adapted in order to embed the bias term
        if  len(self.weights) == 0:
            print("The model is not trained yet")
            return
        
        return np.dot(self.embed_bias(X_test), self.weights)

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
        
    
  