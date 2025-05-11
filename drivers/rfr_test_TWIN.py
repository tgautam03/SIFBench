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
    which_data = "TWIN"
    
    # Specify the directory path
    dir_path = "../files/data/TWIN/{}/TEST".format(which_data)

    # Get all .csv files
    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

    l2_errs_1 = np.zeros((len(csv_files)-5, 3))
    l2_errs_2 = np.zeros((len(csv_files)-5, 3))

    count = 0
    for csv_file in tqdm(csv_files):
        print("=============================")
        print("========= Testing: {} ========".format(csv_file))
        print("=============================")
        # Surface crack only has tension loading
        if "PART_1" in csv_file:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])

            for i in range(2, 7):
                FILE_NAME = csv_file.replace("PART_1", "PART_{}".format(i))
                df_ = pd.read_csv(dir_path + "/{}".format(csv_file))
                df_ = df_.drop(columns=['b/t'])
                df = pd.concat([df, df_], axis=0, ignore_index=True)
            test_combinations = df.iloc[:, 1:7].drop_duplicates().to_numpy()

            # Drop crack index
            d = df.to_numpy()[::8,1:]
            print("Dataset Size: ", d.shape)

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")

            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C1_TENSION.joblib'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")))
            y_pred = rfr.predict(d[:,:7])

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C1_TENSION.npy'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")), y_pred)
            l2_err = mean_normalized_l2(d[:,8].reshape(-1, 16), y_pred.reshape(-1, 16))
            print("L2 error: ", l2_err)

            l2_errs_1[count, 0] = l2_err


            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C2_TENSION.joblib'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")))
            y_pred = rfr.predict(np.delete(d[:,:8], 6, axis=1))

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C2_TENSION.npy'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")), y_pred)
            l2_err = mean_normalized_l2(d[:,9].reshape(-1, 16), y_pred.reshape(-1, 16))
            print("L2 error: ", l2_err)

            l2_errs_2[count, 0] = l2_err

            print("---------------")
            print("Bending Loading")
            print("---------------")
            
            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C1_BENDING.joblib'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")))
            y_pred = rfr.predict(d[:,:7])

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C1_BENDING.npy'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")), y_pred)
            l2_err = mean_normalized_l2(d[:,10].reshape(-1, 16), y_pred.reshape(-1, 16))
            print("L2 error: ", l2_err)

            l2_errs_1[count, 1] = l2_err


            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C2_BENDING.joblib'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")))
            y_pred = rfr.predict(np.delete(d[:,:8], 6, axis=1))

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C2_BENDING.npy'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")), y_pred)
            l2_err = mean_normalized_l2(d[:,11].reshape(-1, 16), y_pred.reshape(-1, 16))
            print("L2 error: ", l2_err)

            l2_errs_2[count, 1] = l2_err

            print("---------------")
            print("Bearing Loading")
            print("---------------")
            
            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C1_BEARING.joblib'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")))
            y_pred = rfr.predict(d[:,:7])

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C1_BEARING.npy'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")), y_pred)
            l2_err = mean_normalized_l2(d[:,12].reshape(-1, 16), y_pred.reshape(-1, 16))
            print("L2 error: ", l2_err)

            l2_errs_1[count, 2] = l2_err


            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C2_BEARING.joblib'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")))
            y_pred = rfr.predict(np.delete(d[:,:8], 6, axis=1))

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C2_BEARING.npy'.format(csv_file[:-4].replace("PART_1_TEST", "TRAIN")), y_pred)
            l2_err = mean_normalized_l2(d[:,13].reshape(-1, 16), y_pred.reshape(-1, 16))
            print("L2 error: ", l2_err)

            l2_errs_2[count, 2] = l2_err
            
            count += 1

        elif "PART_2" in csv_file:
            continue

        elif "PART_3" in csv_file:
            continue

        elif "PART_4" in csv_file:
            continue

        elif "PART_5" in csv_file:
            continue

        elif "PART_6" in csv_file:
            continue

        else:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])
            test_combinations = df.iloc[:, 1:7].drop_duplicates().to_numpy()

            # Drop crack index
            d = df.to_numpy()[::8,1:]
            print("Dataset Size: ", d.shape)

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")

            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C1_TENSION.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = rfr.predict(d[:,:7])

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C1_TENSION.npy'.format(csv_file[:-4]), y_pred)
            # y_pred = np.load('../files/predictions/rfr/{}_TWIN_C1_TENSION.npy'.format(csv_file[:-4]))
            l2_err = mean_normalized_l2(d[:,8].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs_1[count, 0] = l2_err


            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C2_TENSION.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = rfr.predict(np.delete(d[:,:8], 6, axis=1))

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C2_TENSION.npy'.format(csv_file[:-4]), y_pred)
            # y_pred = np.load('../files/predictions/rfr/{}_TWIN_C2_TENSION.npy'.format(csv_file[:-4]))
            l2_err = mean_normalized_l2(d[:,9].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs_2[count, 0] = l2_err

            print("---------------")
            print("Bending Loading")
            print("---------------")
            
            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C1_BENDING.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = rfr.predict(d[:,:7])

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C1_BENDING.npy'.format(csv_file[:-4]), y_pred)
            # y_pred = np.load('../files/predictions/rfr/{}_TWIN_C1_BENDING.npy'.format(csv_file[:-4]))
            l2_err = mean_normalized_l2(d[:,10].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs_1[count, 1] = l2_err


            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C2_BENDING.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = rfr.predict(np.delete(d[:,:8], 6, axis=1))

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C2_BENDING.npy'.format(csv_file[:-4]), y_pred)
            # y_pred = np.load('../files/predictions/rfr/{}_TWIN_C2_BENDING.npy'.format(csv_file[:-4]))
            l2_err = mean_normalized_l2(d[:,11].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs_2[count, 1] = l2_err

            print("---------------")
            print("Bearing Loading")
            print("---------------")
            
            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C1_BEARING.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = rfr.predict(d[:,:7])

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C1_BEARING.npy'.format(csv_file[:-4]), y_pred)
            # y_pred = np.load('../files/predictions/rfr/{}_TWIN_C1_BEARING.npy'.format(csv_file[:-4]))
            l2_err = mean_normalized_l2(d[:,12].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs_1[count, 2] = l2_err


            # Testing
            rfr = load('../files/trained_models/rfr/{}_TWIN_C2_BEARING.joblib'.format(csv_file[:-4].replace("TEST", "TRAIN")))
            y_pred = rfr.predict(np.delete(d[:,:8], 6, axis=1))

            # Saving predictions model
            np.save('../files/predictions/rfr/{}_TWIN_C2_BEARING.npy'.format(csv_file[:-4]), y_pred)
            # y_pred = np.load('../files/predictions/rfr/{}_TWIN_C2_BEARING.npy'.format(csv_file[:-4]))
            l2_err = mean_normalized_l2(d[:,13].reshape(len(test_combinations), 16), y_pred.reshape(len(test_combinations), 16))
            print("L2 error: ", l2_err)

            l2_errs_2[count, 2] = l2_err
            
            count += 1

    np.save("../files/metrics/l2_err_rfr_TWIN_C1.npy", l2_errs_1)
    np.save("../files/metrics/l2_err_rfr_TWIN_C2.npy", l2_errs_2)
