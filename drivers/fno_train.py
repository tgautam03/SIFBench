import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.fno import FNO

from sklearn.model_selection import train_test_split

import itertools
import os

device = 'cuda'
batch_size = 20
lr = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes = 64
width = 64

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
        if "THROUGH" in csv_file:
            print("Skipping ", csv_file)
            continue

        print("=============================")
        print("========= Training: {} ========".format(csv_file))
        print("=============================")
        # Surface crack only has tension loading
        if csv_file[:7] == "SURFACE":
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])
            train_combinations = df.iloc[:, 1:4].drop_duplicates().to_numpy()

            # Drop crack index
            d = df.to_numpy()[:,1:]
            print("Dataset Size: ", d.shape)

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")
            X_fno = np.zeros((len(train_combinations), 128, 4))
            y_fno = np.zeros((len(train_combinations), 128))

            for (i, combination) in enumerate(train_combinations):
                indices = np.where((d[:, 0] == combination[0]) & 
                                (d[:, 1] == combination[1]) &
                                (d[:, 2] == combination[2])) 
                indices = indices[0]

                phi_values = d[indices][:,-2]
                
                X_fno[i,:,:-1] = combination
                X_fno[i,:,-1] = phi_values

                y_fno[i,:] = d[indices][:,-1]


            X_train_gpu = torch.FloatTensor(X_fno[:int(0.8*len(X_fno))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno[int(0.8*len(X_fno)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno[:int(0.8*len(y_fno))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno[int(0.8*len(y_fno)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TENSION".format(csv_file[:-4]))
            print("")

        else:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])
            train_combinations = df.iloc[:, 1:5].drop_duplicates().to_numpy()

            # Drop crack index
            d = df.to_numpy()[:,1:]
            print("Dataset Size: ", d.shape)

            # X, y
            X_fno = np.zeros((len(train_combinations), 128, 5))
            y_fno_tension = np.zeros((len(train_combinations), 128))
            y_fno_bending = np.zeros((len(train_combinations), 128))
            y_fno_bearing = np.zeros((len(train_combinations), 128))

            for (i, combination) in enumerate(train_combinations):
                indices = np.where((d[:, 0] == combination[0]) & 
                                (d[:, 1] == combination[1]) &
                                (d[:, 2] == combination[2]) &
                                (d[:, 3] == combination[3])) 
                indices = indices[0]

                phi_values = d[indices][:,-4]
                
                X_fno[i,:,:-1] = combination
                X_fno[i,:,-1] = phi_values

                y_fno_tension[i,:] = d[indices][:,-3]
                y_fno_bending[i,:] = d[indices][:,-2]
                y_fno_bearing[i,:] = d[indices][:,-1]

            print("---------------")
            print("Tension Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno[:int(0.8*len(X_fno))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno[int(0.8*len(X_fno)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_tension[:int(0.8*len(y_fno_tension))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_tension[int(0.8*len(y_fno_tension)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TENSION".format(csv_file[:-4]))
            print("")

            print("---------------")
            print("Bending Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno[:int(0.8*len(X_fno))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno[int(0.8*len(X_fno)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_bending[:int(0.8*len(y_fno_bending))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_bending[int(0.8*len(y_fno_bending)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_BENDING".format(csv_file[:-4]))
            print("")

            print("---------------")
            print("Bearing Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno[:int(0.8*len(X_fno))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno[int(0.8*len(X_fno)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_bearing[:int(0.8*len(y_fno_bearing))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_bearing[int(0.8*len(y_fno_bearing)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_BEARING".format(csv_file[:-4]))
            print("")