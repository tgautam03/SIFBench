import sys
import pandas as pd
from joblib import dump, load
from tqdm import tqdm
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.nn as custom_nn
from src.fno import FNO, FNO1d

if __name__ == "__main__":
    ##############################################################################
    ############################## Accepting Input ###############################
    ##############################################################################
    if len(sys.argv) > 1:
        what_to_do = sys.argv[1]

    else:
        raise ValueError("Please provide Train or Test as a command-line argument.")
    
    ##############################################################################
    ################################# Training ###################################
    ##############################################################################
    if what_to_do == "Train":
        print("{} using the surface crack dataset".format(what_to_do))

        # Loading dataset
        df_train = pd.read_csv("files/data/SINGLE_CRACK/SURFACE_CRACK/SURFACE_CRACK_TRAIN.csv")
        train_combinations = df_train.iloc[:, 1:4].drop_duplicates().to_numpy()
        d_full = df_train.to_numpy()[:,1:]
        print("Dataset size: {}; Num geom: {}".format(d_full.shape, len(train_combinations)))

        print("---------------")
        print("Tension Loading")
        print("---------------")
        
        ############################### RFR ###############################
        print("==========================================================")
        print("Training RFR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X_train = d[:,:-1]
        y_train = d[:,-1]
        print("Revised input size: {}; Revised output size: {}".format(X_train.shape, y_train.shape))

        rfr = RandomForestRegressor(max_depth=None)
        rfr.fit(X_train, y_train)
        # Saving trained model
        dump(rfr, 'files/trained_models/rfr/{}_TENSION.joblib'.format("SURFACE_CRACK"))
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Training SVR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X_train = d[:,:-1]
        y_train = d[:,-1]
        print("Revised input size: {}; Revised output size: {}".format(X_train.shape, y_train.shape))

        svr = SVR()
        svr.fit(X_train, y_train)
        # Saving trained model
        dump(svr, 'files/trained_models/svr/{}_TENSION.joblib'.format("SURFACE_CRACK"))
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Training NN on a subset of size 500000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 500000), replace=False)
        d = d_full[random_indices]
        
        X_train = d[:,:-1]
        y_train = d[:,-1]
        print("Revised input size: {}; Revised output size: {}".format(X_train.shape, y_train.shape))
        
        device = 'cuda'
        EPOCHS = 150000
        # CPU -> GPU
        X_train_gpu = torch.FloatTensor(X_train[:int(0.8*len(X_train))]).to(device)
        X_val_gpu = torch.FloatTensor(X_train[int(0.8*len(X_train)):]).to(device)
        y_train_gpu = torch.FloatTensor(np.expand_dims(y_train[:int(0.8*len(y_train))], axis=-1)).to(device)
        y_val_gpu = torch.FloatTensor(np.expand_dims(y_train[int(0.8*len(y_train)):], axis=-1)).to(device)
        # Training
        net = custom_nn.Net10(X_train_gpu.shape[1], 15).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        FILENAME = "files/trained_models/nn/{}_TENSION.pt".format("SURFACE_CRACK")
        _, _ = custom_nn.train_model(net, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, loss_fn, optimizer, EPOCHS, FILENAME)
        print("")
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Training FNO on a subset of size 25000 geometries (randomly sampled)")
        random_indices = np.random.choice(len(train_combinations), size=min(len(train_combinations), 25000), replace=False)
        train_combinations = train_combinations[random_indices]

        X_fno = np.zeros((len(train_combinations), 128, 4))
        y_fno = np.zeros((len(train_combinations), 128))
        print("Revised input size: {}; Revised output size: {}".format(X_fno.shape, y_fno.shape))

        for (i, combination) in enumerate(train_combinations):
            indices = np.where((d_full[:, 0] == combination[0]) & 
                            (d_full[:, 1] == combination[1]) &
                            (d_full[:, 2] == combination[2])) 
            indices = indices[0]

            phi_values = d_full[indices][:,-2]
            
            X_fno[i,:,:-1] = combination
            X_fno[i,:,-1] = phi_values

            y_fno[i,:] = d_full[indices][:,-1]

        X_train_gpu = torch.FloatTensor(X_fno[:int(0.8*len(X_fno))]).to(device)
        X_val_gpu = torch.FloatTensor(X_fno[int(0.8*len(X_fno)):]).to(device)

        y_train_gpu = torch.FloatTensor(y_fno[:int(0.8*len(y_fno))]).to(device)
        y_val_gpu = torch.FloatTensor(y_fno[int(0.8*len(y_fno)):]).to(device)

        # Train
        device = 'cuda'
        batch_size = 250000
        lr = 0.001

        epochs = 1000
        step_size = 50
        gamma = 0.5

        modes = 64
        width = 64
        fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
        fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "files/trained_models/fno/", "{}_TENSION".format("SURFACE_CRACK"))

        print("")
    ##############################################################################
    ################################## Testing ###################################
    ##############################################################################
    elif what_to_do == "Test":
        print("{} on the surface crack dataset".format(what_to_do))

        # Loading dataset
        df_test = pd.read_csv("files/data/SINGLE_CRACK/SURFACE_CRACK/SURFACE_CRACK_TEST.csv")
        test_combinations = df_test.iloc[:, 1:4].drop_duplicates().to_numpy()
        d = df_test.to_numpy()[:,1:]
        print("Dataset size: {}; Num geom: {}".format(d.shape, len(test_combinations)))

        print("---------------")
        print("Tension Loading")
        print("---------------")

        ############################### RFR ###############################
        print("==========================================================")
        print("Testing RFR")

        X_test = d[:,:-1]
        y_test = d[:,-1]
        print("Input size: {}; Output size: {}".format(X_test.shape, y_test.shape))

        rfr = load('files/trained_models/rfr/{}_TENSION.joblib'.format("SURFACE_CRACK"))
        y_pred = rfr.predict(X_test)
        # Saving prediction
        np.save('files/predictions/rfr/{}_TENSION.npy'.format("SURFACE_CRACK"), y_pred)
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Testing SVR")
        
        X_test = d[:,:-1]
        y_test = d[:,-1]
        print("Input size: {}; Output size: {}".format(X_test.shape, y_test.shape))

        svr = load('files/trained_models/svr/{}_TENSION.joblib'.format("SURFACE_CRACK"))
        y_pred = svr.predict(X_test)
        # Saving prediction
        np.save('files/predictions/svr/{}_TENSION.npy'.format("SURFACE_CRACK"), y_pred)
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Testing NN")
        device = "cuda"
        X_test = d[:,:-1]
        y_test = d[:,-1]
        print("Input size: {}; Output size: {}".format(X_test.shape, y_test.shape))

        X_test_gpu = torch.FloatTensor(X_test).to(device)
        net = custom_nn.Net10(X_test_gpu.shape[1], 15).to(device)
        FILENAME = 'files/trained_models/nn/{}_TENSION.pt'.format("SURFACE_CRACK")
        net.load_state_dict(torch.load(FILENAME, weights_only=False))
        with torch.no_grad():
            y_pred = net.forward(X_test_gpu).cpu().numpy()[:,0]

        # Saving predictions model
        np.save('files/predictions/nn/{}_TENSION.npy'.format("SURFACE_CRACK"), y_pred)
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Testing FNO")
        X_fno = np.zeros((len(test_combinations), 128, 4))
        y_fno = np.zeros((len(test_combinations), 128))
        print("Input size: {}; Output size: {}".format(X_fno.shape, y_fno.shape))

        for (i, combination) in enumerate(test_combinations):
            indices = np.where((d[:, 0] == combination[0]) & 
                            (d[:, 1] == combination[1]) &
                            (d[:, 2] == combination[2])) 
            indices = indices[0]

            phi_values = d[indices][:,-2]
            
            X_fno[i,:,:-1] = combination
            X_fno[i,:,-1] = phi_values

            y_fno[i,:] = d[indices][:,-1]

        X_test_gpu = torch.FloatTensor(X_fno).to(device)
        y_test_gpu = torch.FloatTensor(y_fno).to(device)

        len_phi_values = 128
        modes = 64
        width = 64
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_TENSION.pt'.format("SURFACE_CRACK"), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_TENSION.pt'.format("SURFACE_CRACK"),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('files/predictions/fno/{}_TENSION.npy'.format("SURFACE_CRACK"), y_pred)

        print("")
         
    ##############################################################################
    ################################## Invalid ###################################
    ##############################################################################
    else:
        raise ValueError("Inputs can only be Train or Test")