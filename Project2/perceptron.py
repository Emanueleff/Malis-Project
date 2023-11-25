from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import time
import csv 

class Perceptron:
    

    def __init__(self, X, y):

        '''
        Initializes the parameters so that the have the same dimensions as the input data + 1
        Inputs:
        X - input data matrix of dimensions N x D => Reshapes the input data so that it can handle w_0, dim: N x (D+1)
        y - assign the label of the training set
        weights - model parameters initialized to zero size (D + 1) x 1
        '''
        self.X = PolynomialFeatures(1).fit_transform(X)
        self.y=y     
        self.weights = np.zeros((X.shape[1]+1))
        

        
    def train(self, alpha):

        while True:
            m = 0

            """
            for i in range (0, self.X.shape[0]):
                if( np.dot(self.X[i,:], self.weights)*self.y[i] <= 0 ):
                    self.weights = self.weights + (alpha)*self.X[i,:].T*self.y[i]
                    m = m + 1
            
            if (m==0):
                return
            """

            result = np.dot(self.X, self.weights)*self.y
            index = np.where(result <= 0)[0]

            if(index.size == 0):
                return
            
            self.weights = self.weights + (alpha)*np.sum(self.X[index,:].T*self.y[index], axis=1)
            
        
    def predict(self, X_test):
        
        X_test = PolynomialFeatures(1).fit_transform(X_test)
        y_hat = np.sign(np.dot(X_test, self.weights)).astype(int)
        
        return y_hat
    
  