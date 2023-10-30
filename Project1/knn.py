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
        distance_mat = self.minkowski_dist(X_new, p)

        dist_ordered = np.argsort(distance_mat, axis=1)
        res = dist_ordered[:, 0:self.k]
        
        label = self.y[0,res[:]]
        
        y_hat = np.count_nonzero( label, axis=1 )
        y_hat = np.where( y_hat < self.k/2 , 0, 1)
        #y_hat[ y_hat < self.k/2 ] = 0
        #y_hat[ y_hat > self.k/2 ] = 1

        """        
        newy = self.y.repeat(480, axis=0)[:, :, np.newaxis]
        X_labelled =  np.concatenate((distance_mat[:,:,np.newaxis], newy), axis=2)
        arr = X_labelled
        idx = np.argsort(arr[...,0], axis=1)
        arr1 = np.take_along_axis(arr, idx[...,None], axis=1)
        res = arr1[:, 0:self.k,1]
        y_hat = np.count_nonzero( res, axis=1 )
        y_hat[ y_hat < self.k/2 ] = 0
        y_hat[ y_hat > self.k/2 ] = 1
        """
    
        return y_hat
    
    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''

        # X_new sono N validation (x1,x2)
        # self.X sono i miei training 
        # ( (X_new_1 - X_1) ^p + (X_new2 - X_2)^p ) ^ (1/p)

        #dst = ((X_new - self.X)**p)**(1/p)

        # reshape( X_new, )
        
        X_3d = self.X[:, :, np.newaxis]
    #    X_3d = np.reshape(X_3d, (1,2,2800) ) #hardcodato pazienza
        #print(X_3d.shape)

        X_new_3d =  np.transpose(X_new)[np.newaxis, :, :] #[X_new[:,0],X_new[0,:] np.newaxis]  #X_new.repeat(X.shape[0], axis=2)    
        X_new_3d = X_new_3d.repeat( self.X.shape[0], axis=0 )
        #print(X_new_3d.shape)


        c = X_new_3d + X_3d
        
        dst =  (( (np.abs(X_new_3d - X_3d)**p).sum(axis=1) ) ** (1/p)).T
        #print(mink)
        #print(mink.shape)

    
        return dst