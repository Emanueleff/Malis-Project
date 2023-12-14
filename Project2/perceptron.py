from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import time
import csv 
import sys

class Perceptron:
    

    def __init__(self, alpha=0.001):

        '''
        Null initialization of the parameters for the model except for alpha that has a defualt value of 0.001
        '''
        self.X = []
        self.y = []
        self.initial_weights = []
        self.weights = []
        self.alpha = alpha

        
    def train(self, X, y, initial_weights=None):

        '''
        Initializes the parameters so that they have the same dimensions as the input data + 1
        Inputs:
        X - input data matrix of dimensions N x D => Reshapes the input data so that it can handle w_0, dim: N x (D+1)
        y - assign the label of the training set
        initial_weights - set of initial weigths with default value to None. If None, the set of weight will be assigned random.
        '''

        self.X = PolynomialFeatures(1).fit_transform(X)
        self.y=y     

        if(initial_weights is None):
            self.initial_weights = np.random.rand(X.shape[1]+1)
            self.weights = self.initial_weights
        else:
            self.initial_weights = initial_weights
            self.weights = initial_weights

        while True:

            #Left as a comment the training algorithm using the for loop
            """
            m = 0
            for i in range (0, self.X.shape[0]):
                if( np.dot(self.X[i,:], self.weights)*self.y[i] <= 0 ):
                    self.weights = self.weights + (alpha)*self.X[i,:].T*self.y[i]
                    m = m + 1
            
            if (m==0):
                return
            """

            #Optimized(?) version without using the for loop

            #vector with the assigned hyperplane for each sample (( N x D+1 )@( D+1 x 1)) = (D+1 x 1)
            
            
            
            
            
            #SHUOLDNT BE N x 1?     
            
            
            
            result = np.dot(self.X, self.weights)*self.y

            #check which sample has negative value, we take the first field of the tuple returned by np.where
            index = np.where(result <= 0)[0]

            #if it has size 0 we have no error and an optimal hyperplane is found
            if(index.size == 0):
                return
            
            #update otherwise, the sum is done over the misclassified sample and their value is summed togheter and then summed to the weight vector
            self.weights = self.weights + (self.alpha)*np.sum(self.X[index,:].T*self.y[index], axis=1)

        
    def predict(self, X_test):

        '''
        Perform Prediction over the test set
        Input:
        X_test - input data matrix of dimensions N x D => Reshapes the input data so that it can handle w_0, dim: N x (D+1)
        Output:
        y_hat: prediction made over the test set
        '''
        
        #increasing size of X_test to handle the bias absorbed in the weight vector
        X_test = PolynomialFeatures(1).fit_transform(X_test)

        #making prediction
        y_hat = np.sign(np.dot(X_test, self.weights)).astype(int)
        
        return y_hat
    
    def computeMargin(self):

        '''
        Compute the margin of the hyperplane
        Output:
        margin: value of the margin
        '''

        #the margin is the minimum distance of a point from the hyperplane
        return np.min(np.abs(np.dot(self.X, self.weights))/np.linalg.norm(self.weights))
    
    def printParams(self):

        '''
        Helper function to know the parameters of the model
        '''

        print("Model Parameters: ")
        print("Initial Weights:", self.initial_weights)
        print("Current Weights:", self.weights)
        print("Learning Rate:", self.alpha)
        
    
  