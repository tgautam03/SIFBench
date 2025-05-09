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
batch_size = 250000
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
        if "PART" in csv_file:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])

            train_combinations = df.iloc[:, 1:7].drop_duplicates().to_numpy()
            random_indices = np.random.choice(len(train_combinations), size=min(len(train_combinations), 30000), replace=False)
            train_combinations = train_combinations[random_indices]

            # Drop crack index
            d = df.to_numpy()[:,1:]
            print("ORIGINAL -> Dataset Size: {}; Num Combinations: {}".format(d.shape, d.shape[0]/128))
            print("REVISED -> Dataset Size: {}; Num Combinations: {}".format(len(train_combinations)*128, len(train_combinations) ))

            X_fno_C1 = np.zeros((len(train_combinations), 128, 7))
            X_fno_C2 = np.zeros((len(train_combinations), 128, 7))

            y_fno_C1_T = np.zeros((len(train_combinations), 128))
            y_fno_C2_T = np.zeros((len(train_combinations), 128))

            y_fno_C1_B = np.zeros((len(train_combinations), 128))
            y_fno_C2_B = np.zeros((len(train_combinations), 128))

            y_fno_C1_P = np.zeros((len(train_combinations), 128))
            y_fno_C2_P = np.zeros((len(train_combinations), 128))

            for (i, combination) in enumerate(train_combinations):
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
            X_train_gpu = torch.FloatTensor(X_fno_C1[:int(0.8*len(X_fno_C1))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C1[int(0.8*len(X_fno_C1)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C1_T[:int(0.8*len(y_fno_C1_T))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C1_T[int(0.8*len(y_fno_C1_T)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C1_TENSION".format(csv_file[:-4]))
            del fno_model


            X_train_gpu = torch.FloatTensor(X_fno_C2[:int(0.8*len(X_fno_C2))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C2[int(0.8*len(X_fno_C2)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C2_T[:int(0.8*len(y_fno_C2_T))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C2_T[int(0.8*len(y_fno_C2_T)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C2_TENSION".format(csv_file[:-4]))
            del fno_model
            print("")

            print("---------------")
            print("BENDING Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno_C1[:int(0.8*len(X_fno_C1))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C1[int(0.8*len(X_fno_C1)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C1_B[:int(0.8*len(y_fno_C1_B))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C1_B[int(0.8*len(y_fno_C1_B)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C1_BENDING".format(csv_file[:-4]))
            del fno_model


            X_train_gpu = torch.FloatTensor(X_fno_C2[:int(0.8*len(X_fno_C2))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C2[int(0.8*len(X_fno_C2)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C2_B[:int(0.8*len(y_fno_C2_B))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C2_B[int(0.8*len(y_fno_C2_B)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C2_BENDING".format(csv_file[:-4]))
            del fno_model
            print("")

            print("---------------")
            print("BEARING Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno_C1[:int(0.8*len(X_fno_C1))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C1[int(0.8*len(X_fno_C1)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C1_P[:int(0.8*len(y_fno_C1_P))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C1_P[int(0.8*len(y_fno_C1_P)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C1_BEARING".format(csv_file[:-4]))
            del fno_model


            X_train_gpu = torch.FloatTensor(X_fno_C2[:int(0.8*len(X_fno_C2))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C2[int(0.8*len(X_fno_C2)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C2_P[:int(0.8*len(y_fno_C2_P))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C2_P[int(0.8*len(y_fno_C2_P)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C2_BEARING".format(csv_file[:-4]))
            del fno_model
            print("")

        else:
            df = pd.read_csv(dir_path + "/{}".format(csv_file))
            df = df.drop(columns=['b/t'])
            train_combinations = df.iloc[:, 1:7].drop_duplicates().to_numpy()
            random_indices = np.random.choice(len(train_combinations), size=min(30000, len(train_combinations)), replace=False)
            train_combinations = train_combinations[random_indices]

            # Drop crack index
            d = df.to_numpy()[:,1:]
            print("ORIGINAL -> Dataset Size: {}; Num Combinations: {}".format(d.shape, d.shape[0]/128))
            print("REVISED -> Dataset Size: {}; Num Combinations: {}".format(len(train_combinations)*128, len(train_combinations) ))

            # X, y
            print("---------------")
            print("Tension Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno_C1[:int(0.8*len(X_fno_C1))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C1[int(0.8*len(X_fno_C1)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C1_T[:int(0.8*len(y_fno_C1_T))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C1_T[int(0.8*len(y_fno_C1_T)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C1_TENSION".format(csv_file[:-4]))
            del fno_model


            X_train_gpu = torch.FloatTensor(X_fno_C2[:int(0.8*len(X_fno_C2))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C2[int(0.8*len(X_fno_C2)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C2_T[:int(0.8*len(y_fno_C2_T))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C2_T[int(0.8*len(y_fno_C2_T)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C2_TENSION".format(csv_file[:-4]))
            del fno_model
            print("")

            print("---------------")
            print("BENDING Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno_C1[:int(0.8*len(X_fno_C1))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C1[int(0.8*len(X_fno_C1)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C1_B[:int(0.8*len(y_fno_C1_B))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C1_B[int(0.8*len(y_fno_C1_B)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C1_BENDING".format(csv_file[:-4]))
            del fno_model


            X_train_gpu = torch.FloatTensor(X_fno_C2[:int(0.8*len(X_fno_C2))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C2[int(0.8*len(X_fno_C2)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C2_B[:int(0.8*len(y_fno_C2_B))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C2_B[int(0.8*len(y_fno_C2_B)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C2_BENDING".format(csv_file[:-4]))
            del fno_model
            print("")

            print("---------------")
            print("BEARING Loading")
            print("---------------")
            X_train_gpu = torch.FloatTensor(X_fno_C1[:int(0.8*len(X_fno_C1))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C1[int(0.8*len(X_fno_C1)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C1_P[:int(0.8*len(y_fno_C1_P))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C1_P[int(0.8*len(y_fno_C1_P)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C1_BEARING".format(csv_file[:-4]))
            del fno_model


            X_train_gpu = torch.FloatTensor(X_fno_C2[:int(0.8*len(X_fno_C2))]).to(device)
            X_val_gpu = torch.FloatTensor(X_fno_C2[int(0.8*len(X_fno_C2)):]).to(device)

            y_train_gpu = torch.FloatTensor(y_fno_C2_P[:int(0.8*len(y_fno_C2_P))]).to(device)
            y_val_gpu = torch.FloatTensor(y_fno_C2_P[int(0.8*len(y_fno_C2_P)):]).to(device)

            # Train
            fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
            fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "../files/trained_models/fno/", "{}_TWIN_C2_BEARING".format(csv_file[:-4]))
            del fno_model
            print("")