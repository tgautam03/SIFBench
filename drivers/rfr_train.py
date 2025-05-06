import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import itertools
import os

def shuffle_arrays(arr1, arr2):
    assert len(arr1) == len(arr2)
    permutation = np.random.permutation(len(arr1))
    return arr1[permutation], arr2[permutation]

if __name__ == "__main__":
    # SINGLE or TWIN
    which_data = "SINGLE"
    
    # Specify the directory path
    dir_path = "../files/data/FINAL_CSV/{}/TRAIN".format(which_data)

    # Get all .csv files
    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files):
        print("=============================")
        print("========= Training: {} ========".format(csv_file))
        print("=============================")
        # Surface crack only has tension loading
        if csv_file[:7] == "SURFACE":
            df = pd.read_csv(dir_path + "/{}".format(csv_file))

            # Drop crack index
            d = df.to_numpy()[:,1:]
            print("Dataset Size: ", d.shape)
            print("Training on a subset of size 100000 (randomly sampled)")
            random_indices = np.random.choice(len(d), size=100000, replace=False)
            d = d[random_indices]
            print("Revised dataset Size: ", d.shape)

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")
            X_train, y_train = shuffle_arrays(d[:,:-1], d[:,-1])

            # Train
            rfr = RandomForestRegressor(max_depth=None)
            rfr.fit(X_train, y_train)

            # Saving trained model
            dump(rfr, '../files/trained_models/rfr/{}_TENSION.joblib'.format(csv_file[:-4]))
            print("")

        else:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])

            # Drop crack index
            d = df.to_numpy()[:,1:]
            print("Dataset Size: ", d.shape)
            print("Training on a subset of size 100000 (randomly sampled)")
            random_indices = np.random.choice(len(d), size=100000, replace=False)
            d = d[random_indices]
            print("Revised dataset Size: ", d.shape)

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")
            X_train, y_train = shuffle_arrays(d[:,:-3], d[:,-3])

            # Train
            rfr = RandomForestRegressor(max_depth=None)
            rfr.fit(X_train, y_train)

            # Saving trained model
            dump(rfr, '../files/trained_models/rfr/{}_TENSION.joblib'.format(csv_file[:-4]))
            print("")

            print("---------------")
            print("Bending Loading")
            print("---------------")
            X_train, y_train = shuffle_arrays(d[:,:-3], d[:,-2])

            # Train
            rfr = RandomForestRegressor(max_depth=None)
            rfr.fit(X_train, y_train)

            # Saving trained model
            dump(rfr, '../files/trained_models/rfr/{}_BENDING.joblib'.format(csv_file[:-4]))
            print("")

            print("---------------")
            print("Bearing Loading")
            print("---------------")
            X_train, y_train = shuffle_arrays(d[:,:-3], d[:,-1])

            # Train
            rfr = RandomForestRegressor(max_depth=None)
            rfr.fit(X_train, y_train)

            # Saving trained model
            dump(rfr, '../files/trained_models/rfr/{}_BEARING.joblib'.format(csv_file[:-4]))
            print("")
