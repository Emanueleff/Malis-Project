from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import time
import csv 

class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''   
        self.X=X
        self.y=y     
       
    def predict(self,X_new,p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        ''' 
        # Creating distance matrix
        distance_mat = self.minkowski_dist(X_new, p)

        # Sorting the distance matrix for the first k position
        dist_ordered = np.argpartition(distance_mat, self.k, axis=1) 
        # Take only the sorted part of the distance 
        res = dist_ordered[:, 0:self.k]

        # Take the label of the k-nearest neighbors
        label = self.y[0,res[:]]
        
        # Counting the number of element != 0 and setting 0 / 1 if more than half of the votes are 1 / 0 
        y_hat = np.count_nonzero( label, axis=1 )
        y_hat = np.where( y_hat < self.k/2 , 0, 1)
        
        return y_hat
    
    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        # Reshaping X adding a 3 dimension in order to use broadcasting [NxDx1]
        X_3d = self.X[:, :, np.newaxis]                                     # [N x D x 1]
        
        # Reshaping X_new adding a 3 dimension and repeating the data in this dimension
        X_new_3d =  np.transpose(X_new)[np.newaxis, :, :]                   # [1 X D X M]

        # Calculating the distance matrix using the minkowski formula with parameter p
        dst =  (( (np.abs(X_new_3d - X_3d)**p).sum(axis=1) ) ** (1/p)).T    # [M X N]
        # [1 X D X M] - [N x D x 1] = [N x D x M]
        # [N x D x M].sum(axis=1) = [N x M] -> .t -> [M x N]

        return dst
    
def main(mode):
    # Importing training set
    training_set = pd.read_csv("./training.csv")
    X = training_set[["X1","X2"]].values
    y = training_set["y"].values.astype(int)

    # Importing validation set
    validation_set = pd.read_csv("./validation.csv")
    X_new = validation_set[["X1","X2"]].values
    y_new = validation_set["y"].values.astype(int)

    # Adjusting y shape in order to have 1D array
    y = np.reshape(y, (y.size,1)).T
    y_new = np.reshape(y_new, (y_new.size,1)).T
    
    if (mode == "tuning"):
        datas = []
        max = [0,0,0,0]
        
        # Searching for best k and p 
        for k in range(1,100):
            for p in range(1,11):

                # Declaring class and training the model            
                model = KNN(k)
                model.train(X,y)

                # Testing the model on the validation set
                start_time = time.time()
                y_hat = model.predict(X_new, p)
                end_time = time.time()
                execution_time = end_time - start_time

                # Calculating accuracy of the model with parameters k and p            
                acc = np.sum(y_hat == y_new)/480 * 100
                datas.append([k, p, acc, execution_time])
                # print(f"k={k} p={p}  accuracy={acc}")

                # Utility to know which are the hyperparams for the max accuracy 
                if acc>max[2]:
                    max[0] = k
                    max[1] = p
                    max[2] = acc
                    max[3] = execution_time
        print(f"Max {max}")
        
        # Writing the results in a .csv
        csv_filename = 'results.csv'
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['k', 'p', 'accuracy', 'time'])
            for row in datas:
                writer.writerow(row)
    elif mode == "best": 
        k = 23
        p = 5
        model = KNN(k)
        model.train(X,y)

        # Testing the model on the validation set
        start_time = time.time()
        y_hat = model.predict(X_new, p)
        end_time = time.time()
        execution_time = end_time - start_time
        acc = np.sum(y_hat == y_new)/X_new.shape[0] * 100
        print(f"time: {execution_time}, accuracy: {acc}")
    else: 
        print("Give a parameter 'tuning' or 'best'")
    
if __name__ == "__main__":
    #main("tuning")
    main("best")