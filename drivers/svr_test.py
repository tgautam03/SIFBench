import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm

import itertools
import os

def mean_normalized_l2(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true -  y_pred, ord=2, axis=1) / np.linalg.norm(y_true, ord=2, axis=1))

if __name__ == "__main__":
    # SINGLE or TWIN
    which_data = "SINGLE"
    
    # Specify the directory path
    dir_path = "../files/data/FINAL_CSV/{}/TEST".format(which_data)

    # Get all .csv files
    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

    l2_errs = np.zeros((len(csv_files), 3))

    count = 0
    for csv_file in tqdm(csv_files):
        print("=============================")
        print("========= Testing: {} ========".format(csv_file))
        print("=============================")
        # Surface crack only has tension loading
        if csv_file[:7] == "SURFACE":
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            test_combinations = df.iloc[:, 1:4].drop_duplicates().to_numpy()

            # Drop crack index
            d = df.to_numpy()[::8,1:]
            print("Dataset Size: ", d.shape)

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")

            # Testing
            svr = load('../files/trained_models/svr/{}_TENSION.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = svr.predict(d[:,:-1])

            # Saving predictions model
            np.save('../files/predictions/svr/{}_TENSION.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(d[:,-1].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs[count, 0] = l2_err
            count += 1

        else:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])
            test_combinations = df.iloc[:, 1:5].drop_duplicates().to_numpy()

            # Drop crack index
            d = df.to_numpy()[::8,1:]
            print("Dataset Size: ", d.shape)

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")

            # Testing
            svr = load('../files/trained_models/svr/{}_TENSION.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = svr.predict(d[:,:-3])

            # Saving predictions model
            np.save('../files/predictions/svr/{}_TENSION.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(d[:,-3].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs[count, 0] = l2_err

            print("---------------")
            print("Bending Loading")
            print("---------------")
            
            # Testing
            svr = load('../files/trained_models/svr/{}_BENDING.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = svr.predict(d[:,:-3])

            # Saving predictions model
            np.save('../files/predictions/svr/{}_BENDING.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(d[:,-2].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs[count, 1] = l2_err

            print("---------------")
            print("Bearing Loading")
            print("---------------")
            
            # Testing
            svr = load('../files/trained_models/svr/{}_BEARING.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = svr.predict(d[:,:-3])

            # Saving predictions model
            np.save('../files/predictions/svr/{}_BEARING.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(d[:,-1].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs[count, 2] = l2_err
            
            count += 1

    np.save("../files/metrics/l2_err_svr.npy", l2_errs)
