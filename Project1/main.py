import knn
import utils
import pandas as pd
import os
import numpy as np
import time
import csv 

def main():
    training_set = pd.read_csv("./training.csv")
    X = training_set[["X1","X2"]].values
    y = training_set["y"].values.astype(int)

    y = np.reshape(y, (y.size,1)).T

    validation_set = pd.read_csv("./validation.csv")
    X_new = validation_set[["X1","X2"]].values
    y_new = validation_set["y"].values.astype(int)

    
    y_new = np.reshape(y_new, (y_new.size,1)).T
    datas = []
    max = [0,0,0,0]
    for k in range(1,100):
        for p in range(1,11):
            
            model = knn.KNN(k)
            model.train(X,y)
            start_time = time.time()
            y_hat = model.predict(X_new, p)
            end_time = time.time()
            execution_time = end_time - start_time
            
            acc = np.sum(y_hat == y_new)/480 * 100
            #print(f"k={k} p={p}  accuracy={acc}")
            datas.append([k, p, acc, execution_time]) 
            if acc>max[2]:
                max[0] = k
                max[1] = p
                max[2] = acc
                max[3] = execution_time
    print(f"Max {max}")
    
    
    
    csv_filename = 'results.csv'
    # Apri il file in modalità di scrittura
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Scrivi l'intestazione del file CSV
        writer.writerow(['k', 'p', 'accuracy', 'time'])
        # Scrivi i dati
        for row in datas:
            writer.writerow(row)
    

if __name__ == "__main__":
    main()