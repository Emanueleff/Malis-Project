import knn
import utils
import pandas as pd
import os
import numpy as np
import time
import csv 

def main():
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
    
    datas = []
    max = [0,0,0,0]
    # Searching for best k and p 
    for k in range(1,100):
        for p in range(1,11):

            # Declaring class and training the model            
            model = knn.KNN(k)
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
    
if __name__ == "__main__":
    main()