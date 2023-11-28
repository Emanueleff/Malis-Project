from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import time
import csv 
import sys

class Perceptron:
    

    def __init__(self, alpha=0.001, iter=None):

        '''
        Null initialization of the parameters for the model
        '''
        self.X = []
        self.y = []
        self.weights = []
        self.alpha = alpha
        self.iter = iter

        
    def train(self, X, y, randomize_weigths=True):

        '''
        Initializes the parameters so that the have the same dimensions as the input data + 1
        Inputs:
        X - input data matrix of dimensions N x D => Reshapes the input data so that it can handle w_0, dim: N x (D+1)
        y - assign the label of the training set
        weights - model parameters initialized to zero size (D + 1) x 1
        alpha - learning rate (default value of 0.001)
        iter - max num of iteration (defualt value at 1000)
        
        '''

        self.X = PolynomialFeatures(1).fit_transform(X)
        self.y=y     

        if(randomize_weigths):
            self.weights = np.random.rand(X.shape[1]+1)
        else:
            self.weights = np.zeros((X.shape[1]+1))

        if(self.iter == None):
            while True:
                """
                m = 0
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
                
                self.weights = self.weights + (self.alpha)*np.sum(self.X[index,:].T*self.y[index], axis=1)
        else:
            for i in range(0, self.iter):
                result = np.dot(self.X, self.weights)*self.y
                index = np.where(result <= 0)[0]

                if(index.size == 0):
                    return
                
                self.weights = self.weights + (self.alpha)*np.sum(self.X[index,:].T*self.y[index], axis=1)

        
            
        
    def predict(self, X_test):
        
        X_test = PolynomialFeatures(1).fit_transform(X_test)
        y_hat = np.sign(np.dot(X_test, self.weights)).astype(int)
        
        return y_hat
    
    def computeMargin(self):

        return np.min(np.abs(np.dot(self.X, self.weights)))
        
    
  