from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import time
import csv 

class Perceptron:
    

    def __init__(self):
        
        # empty initialization of X and y
        self.X = []
        self.y = []
        self.w = 0
        
    def train(self,X,y):
        
        self.X=X
        self.y=y     
       
    def predict(self,X_new,p):
        
        
        return 0    
    
    