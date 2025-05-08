import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

import itertools
import os

def shuffle_arrays(arr1, arr2):
    assert len(arr1) == len(arr2)
    permutation = np.random.permutation(len(arr1))
    return arr1[permutation], arr2[permutation]

if __name__ == "__main__":
    # SINGLE or TWIN
    which_data = "TWIN"
    
    # Specify the directory path
    dir_path = "../files/data/FINAL_CSV/{}/TRAIN".format(which_data)

    # Get all .csv files
    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files):
        print("=============================")
        print("========= Training: {} ========".format(csv_file))
        print("=============================")
        # Surface crack only has tension loading
        if "CS2_QUARTER_ELLIPSE" in csv_file:
            continue

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
            X_train, y_train = shuffle_arrays(d[:,:7], d[:,8])

            # Train
            svr = SVR()
            svr.fit(X_train, y_train)

            # Saving trained model
            dump(svr, '../files/trained_models/svr/{}_TWIN_C1_TENSION.joblib'.format(csv_file[:-4]))



            X_train, y_train = shuffle_arrays(np.delete(d[:,:8], 6, axis=1), d[:,9])

            # Train
            svr = SVR()
            svr.fit(X_train, y_train)

            # Saving trained model
            dump(svr, '../files/trained_models/svr/{}_TWIN_C2_TENSION.joblib'.format(csv_file[:-4]))
            print("")

            print("---------------")
            print("Bending Loading")
            print("---------------")
            X_train, y_train = shuffle_arrays(d[:,:7], d[:,10])

            # Train
            svr = SVR()
            svr.fit(X_train, y_train)

            # Saving trained model
            dump(svr, '../files/trained_models/svr/{}_TWIN_C1_BENDING.joblib'.format(csv_file[:-4]))



            X_train, y_train = shuffle_arrays(np.delete(d[:,:8], 6, axis=1), d[:,11])

            # Train
            svr = SVR()
            svr.fit(X_train, y_train)

            # Saving trained model
            dump(svr, '../files/trained_models/svr/{}_TWIN_C2_BENDING.joblib'.format(csv_file[:-4]))
            print("")

            print("---------------")
            print("Bearing Loading")
            print("---------------")
            X_train, y_train = shuffle_arrays(d[:,:7], d[:,12])

            # Train
            svr = SVR()
            svr.fit(X_train, y_train)

            # Saving trained model
            dump(svr, '../files/trained_models/svr/{}_TWIN_C1_BEARING.joblib'.format(csv_file[:-4]))



            X_train, y_train = shuffle_arrays(np.delete(d[:,:8], 6, axis=1), d[:,13])

            # Train
            svr = SVR()
            svr.fit(X_train, y_train)

            # Saving trained model
            dump(svr, '../files/trained_models/svr/{}_TWIN_C2_BEARING.joblib'.format(csv_file[:-4]))
            print("")
