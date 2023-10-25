import knn
import utils
import pandas as pd
import os
import numpy as np

def main():
    training_set = pd.read_csv("./Project1/training.csv")
    X = training_set[["X1","X2"]].values
    y = training_set["y"].values.astype(int)

    y = np.reshape(y, (y.size,1))

    validation_set = pd.read_csv("./Project1/validation.csv")
    X_new = validation_set[["X1","X2"]].values
    y_new = validation_set["y"].values.astype(int)

    
    y_new = np.reshape(y_new, (y_new.size,1))

    """model = knn.KNN(1)

    model.train(X,y)
    model.minkowski_dist(X_new,1)"""



    

    
    

if __name__ == "__main__":
    main()