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
            d = df.to_numpy()[:,1:]
            print("Dataset Size: ", d.shape)

            X_fno = np.zeros((len(test_combinations), 128, 4))
            y_fno = np.zeros((len(test_combinations), 128))

            for (i, combination) in enumerate(test_combinations):
                indices = np.where((d[:, 0] == combination[0]) & 
                                (d[:, 1] == combination[1]) &
                                (d[:, 2] == combination[2])) 
                indices = indices[0]
                
                X_fno[i,:,:-1] = combination
                X_fno[i,:,-1] = d[indices][:, -2]

                y_fno[i,:] = d[indices][:,-1]

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")

            # Testing
            X_test_gpu = torch.FloatTensor(X_fno).to(device)
            y_test_gpu = torch.FloatTensor(y_fno).to(device)

            len_phi_values = 128
            fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
            if device == "cpu":
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
            else:
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
            with torch.no_grad():
                y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

            # Saving predictions model
            np.save('../files/predictions/fno/{}_TENSION.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(y_fno, y_pred)
            print("L2 error: ", l2_err)

            l2_errs[count, 0] = l2_err
            count += 1

        else:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])
            test_combinations = df.iloc[:, 1:5].drop_duplicates().to_numpy()

            # Drop crack index
            d = df.to_numpy()[:,1:]
            print("Dataset Size: ", d.shape)

            X_fno = np.zeros((len(test_combinations), 128, 5))
            y_fno_t = np.zeros((len(test_combinations), 128))
            y_fno_b = np.zeros((len(test_combinations), 128))
            y_fno_p = np.zeros((len(test_combinations), 128))

            for (i, combination) in enumerate(test_combinations):
                indices = np.where((d[:, 0] == combination[0]) & 
                                (d[:, 1] == combination[1]) &
                                (d[:, 2] == combination[2]) &
                                (d[:, 3] == combination[3])) 
                indices = indices[0]
                
                X_fno[i,:,:-1] = combination
                X_fno[i,:,-1] = d[indices][:, -4]

                y_fno_t[i,:] = d[indices][:,-3]
                y_fno_b[i,:] = d[indices][:,-2]
                y_fno_p[i,:] = d[indices][:,-1]

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")

            # Testing
            X_test_gpu = torch.FloatTensor(X_fno).to(device)
            y_test_gpu = torch.FloatTensor(y_fno_t).to(device)

            len_phi_values = 128
            fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
            if device == "cpu":
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
            else:
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_TENSION.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
            with torch.no_grad():
                y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

            # Saving predictions model
            np.save('../files/predictions/fno/{}_TENSION.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(y_fno_t, y_pred)
            print("L2 error: ", l2_err)

            l2_errs[count, 0] = l2_err

            print("---------------")
            print("Bending Loading")
            print("---------------")
            
            # Testing
            X_test_gpu = torch.FloatTensor(X_fno).to(device)
            y_test_gpu = torch.FloatTensor(y_fno_b).to(device)

            len_phi_values = 128
            fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
            if device == "cpu":
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_BENDING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
            else:
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_BENDING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
            with torch.no_grad():
                y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

            # Saving predictions model
            np.save('../files/predictions/fno/{}_BENDING.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(y_fno_b, y_pred)
            print("L2 error: ", l2_err)

            l2_errs[count, 1] = l2_err

            print("---------------")
            print("Bearing Loading")
            print("---------------")
            
            # Testing
            X_test_gpu = torch.FloatTensor(X_fno).to(device)
            y_test_gpu = torch.FloatTensor(y_fno_p).to(device)

            len_phi_values = 128
            fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
            if device == "cpu":
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_BEARING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")), map_location=torch.device('cpu'), weights_only=False))
            else:
                fno_loaded_model.load_state_dict(torch.load('../files/trained_models/fno/fno_{}_BEARING.pt'.format(csv_file[:-4].replace("TEST", "TRAIN")),weights_only=False))
            with torch.no_grad():
                y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

            # Saving predictions model
            np.save('../files/predictions/fno/{}_BEARING.npy'.format(csv_file[:-4]), y_pred)
            l2_err = mean_normalized_l2(y_fno_p, y_pred)
            print("L2 error: ", l2_err)

            l2_errs[count, 2] = l2_err
            
            count += 1

    np.save("../files/metrics/l2_err_fno.npy", l2_errs)
