from scipy.spatial import distance_matrix
import numpy as np

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