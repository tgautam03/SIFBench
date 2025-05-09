import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.fno import FNO1d

import itertools
import os

modes = 64
width = 64
device = 'cuda'

def mean_normalized_l2(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true -  y_pred, ord=2, axis=1) / np.linalg.norm(y_true, ord=2, axis=1))

if __name__ == "__main__":
    # SINGLE or TWIN
    which_data = "TWIN"
    
    # Specify the directory path
    dir_path = "../files/data/FINAL_CSV/{}/TEST".format(which_data)

    # Get all .csv files
    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

    l2_errs_1 = np.zeros((len(csv_files), 3))
    l2_errs_2 = np.zeros((len(csv_files), 3))

    count = 0
    for csv_file in tqdm(csv_files):
        print("=============================")
        print("========= Testing: {} ========".format(csv_file))
        print("=============================")
        df = pd.read_csv(dir_path + "/{}".format(csv_file))
        df = df.drop(columns=['b/t'])
        test_combinations = df.iloc[:, 1:7].drop_duplicates().to_numpy()

        # Drop crack index
        d = df.to_numpy()[:,1:]
        print("Dataset Size: ", d.shape)

        X_fno_C1 = np.zeros((len(test_combinations), 128, 7))
        X_fno_C2 = np.zeros((len(test_combinations), 128, 7))

        y_fno_C1_T = np.zeros((len(test_combinations), 128))
        y_fno_C2_T = np.zeros((len(test_combinations), 128))

        y_fno_C1_B = np.zeros((len(test_combinations), 128))
        y_fno_C2_B = np.zeros((len(test_combinations), 128))

        y_fno_C1_P = np.zeros((len(test_combinations), 128))
        y_fno_C2_P = np.zeros((len(test_combinations), 128))

        for (i, combination) in enumerate(test_combinations):
            indices = np.where((d[:, 0] == combination[0]) & 
                            (d[:, 1] == combination[1]) &
                            (d[:, 2] == combination[2]) &
                            (d[:, 3] == combination[3]) &
                            (d[:, 4] == combination[4]) &
                            (d[:, 5] == combination[5])) 
            indices = indices[0]

            phi_values = d[indices][:,6]
            X_fno_C1[i,:,:-1] = combination
            X_fno_C1[i,:,-1] = phi_values

            phi_values = d[indices][:,7]
            X_fno_C2[i,:,:-1] = combination
            X_fno_C2[i,:,-1] = phi_values

            y_fno_C1_T[i,:] = d[indices][:,8]
            y_fno_C2_T[i,:] = d[indices][:,9]

            y_fno_C1_B[i,:] = d[indices][:,10]
            y_fno_C2_B[i,:] = d[indices][:,11]

            y_fno_C1_P[i,:] = d[indices][:,12]
            y_fno_C2_P[i,:] = d[indices][:,13]

        del(d)

        # X, y
        print("---------------")
        print("Tension Loading")
        print("---------------")
        X_test_gpu = torch.FloatTensor(X_fno_C1).to(device)
        y_test_gpu = torch.FloatTensor(y_fno_C1_T).to(device)

        len_phi_values = 128
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C1_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C1_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('../files/predictions/fno/{}_TWIN_C1_TENSION.npy'.format(csv_file[:-4]), y_pred)
        l2_errs_1[count, 0] = mean_normalized_l2(y_fno_C1_T, y_pred)
        print("L2 error: ", l2_errs_1[count, 0])



        X_test_gpu = torch.FloatTensor(X_fno_C2).to(device)
        y_test_gpu = torch.FloatTensor(y_fno_C2_T).to(device)

        len_phi_values = 128
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C2_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C2_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('../files/predictions/fno/{}_TWIN_C2_TENSION.npy'.format(csv_file[:-4]), y_pred)
        l2_errs_2[count, 0] = mean_normalized_l2(y_fno_C2_T, y_pred)
        print("L2 error: ", l2_errs_2[count, 0])

        print("---------------")
        print("Bending Loading")
        print("---------------")
        X_test_gpu = torch.FloatTensor(X_fno_C1).to(device)
        y_test_gpu = torch.FloatTensor(y_fno_C1_B).to(device)

        len_phi_values = 128
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C1_BENDING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C1_BENDING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('../files/predictions/fno/{}_TWIN_C1_BENDING.npy'.format(csv_file[:-4]), y_pred)
        l2_errs_1[count, 1] = mean_normalized_l2(y_fno_C1_B, y_pred)
        print("L2 error: ", l2_errs_1[count, 1])



        X_test_gpu = torch.FloatTensor(X_fno_C2).to(device)
        y_test_gpu = torch.FloatTensor(y_fno_C2_B).to(device)

        len_phi_values = 128
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C2_BENDING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C2_BENDING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('../files/predictions/fno/{}_TWIN_C2_BENDING.npy'.format(csv_file[:-4]), y_pred)
        l2_errs_2[count, 1] = mean_normalized_l2(y_fno_C2_B, y_pred)
        print("L2 error: ", l2_errs_2[count, 1])

        print("---------------")
        print("Bearing Loading")
        print("---------------")
        X_test_gpu = torch.FloatTensor(X_fno_C1).to(device)
        y_test_gpu = torch.FloatTensor(y_fno_C1_P).to(device)

        len_phi_values = 128
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C1_BEARING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C1_BEARING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('../files/predictions/fno/{}_TWIN_C1_BEARING.npy'.format(csv_file[:-4]), y_pred)
        l2_errs_1[count, 2] = mean_normalized_l2(y_fno_C1_P, y_pred)
        print("L2 error: ", l2_errs_1[count, 2])



        X_test_gpu = torch.FloatTensor(X_fno_C2).to(device)
        y_test_gpu = torch.FloatTensor(y_fno_C2_P).to(device)

        len_phi_values = 128
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C2_BEARING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TWIN_C2_BEARING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('../files/predictions/fno/{}_TWIN_C2_BEARING.npy'.format(csv_file[:-4]), y_pred)
        l2_errs_2[count, 2] = mean_normalized_l2(y_fno_C2_P, y_pred)
        print("L2 error: ", l2_errs_2[count, 2])
        
        count += 1

    np.save("../files/metrics/l2_err_fno_TWIN_C1.npy", l2_errs_1)
    np.save("../files/metrics/l2_err_fno_TWIN_C2.npy", l2_errs_2)
