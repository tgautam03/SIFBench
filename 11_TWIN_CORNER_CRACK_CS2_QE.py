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
        print("{} using the quarter-ellipse corner crack at countersunk hole 2 dataset".format(what_to_do))

        # Loading dataset
        df_train = pd.read_csv("files/data/TWIN/CORNER_CRACK_COUNTERSUNK_HOLE/TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1_TRAIN.csv")
        df_train = df_train.drop(columns=['b/t'])
        df_train = df_train[(df_train["W/R"] >= 4) & (df_train["W/R"] <= 10)]
        df_train = df_train[(df_train["r/t"] >= 0.5) & (df_train["r/t"] <= 1.5)]
        train_combinations = df_train.iloc[:, 1:7].drop_duplicates().to_numpy()
        d_full = df_train.to_numpy()[:,1:]
        print("Dataset size: {}; Num geom: {}".format(d_full.shape, len(train_combinations)))

        print("---------------")
        print("TENSION Loading")
        print("---------------")
        
        ############################### RFR ###############################
        print("==========================================================")
        print("Training RFR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X1_train = d[:,:7]
        y1_train = d[:,8]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))

        rfr = RandomForestRegressor(max_depth=None)
        rfr.fit(X1_train, y1_train)
        # Saving trained model
        dump(rfr, 'files/trained_models/rfr/{}_C1_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,9]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        rfr = RandomForestRegressor(max_depth=None)
        rfr.fit(X2_train, y2_train)
        # Saving trained model
        dump(rfr, 'files/trained_models/rfr/{}_C2_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Training SVR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X1_train = d[:,:7]
        y1_train = d[:,8]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))

        svr = SVR()
        svr.fit(X1_train, y1_train)
        # Saving trained model
        dump(svr, 'files/trained_models/svr/{}_C1_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,9]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        svr = SVR()
        svr.fit(X2_train, y2_train)
        # Saving trained model
        dump(svr, 'files/trained_models/svr/{}_C2_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Training NN on a subset of size 500000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 500000), replace=False)
        d = d_full[random_indices]
        
        X1_train = d[:,:7]
        y1_train = d[:,8]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))
        
        device = 'cuda'
        EPOCHS = 150000
        # CPU -> GPU
        X_train_gpu = torch.FloatTensor(X1_train[:int(0.8*len(X1_train))]).to(device)
        X_val_gpu = torch.FloatTensor(X1_train[int(0.8*len(X1_train)):]).to(device)
        y_train_gpu = torch.FloatTensor(np.expand_dims(y1_train[:int(0.8*len(y1_train))], axis=-1)).to(device)
        y_val_gpu = torch.FloatTensor(np.expand_dims(y1_train[int(0.8*len(y1_train)):], axis=-1)).to(device)
        # Training
        net = custom_nn.Net10(X_train_gpu.shape[1], 15).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        FILENAME = "files/trained_models/nn/{}_C1_TENSION.pt".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        _, _ = custom_nn.train_model(net, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, loss_fn, optimizer, EPOCHS, FILENAME)
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,9]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        device = 'cuda'
        EPOCHS = 150000
        # CPU -> GPU
        X_train_gpu = torch.FloatTensor(X2_train[:int(0.8*len(X2_train))]).to(device)
        X_val_gpu = torch.FloatTensor(X2_train[int(0.8*len(X2_train)):]).to(device)
        y_train_gpu = torch.FloatTensor(np.expand_dims(y2_train[:int(0.8*len(y2_train))], axis=-1)).to(device)
        y_val_gpu = torch.FloatTensor(np.expand_dims(y2_train[int(0.8*len(y2_train)):], axis=-1)).to(device)
        # Training
        net = custom_nn.Net10(X_train_gpu.shape[1], 15).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        FILENAME = "files/trained_models/nn/{}_C2_TENSION.pt".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        _, _ = custom_nn.train_model(net, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, loss_fn, optimizer, EPOCHS, FILENAME)
        print("")
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Training FNO on a subset of size 25000 geometries (randomly sampled)")
        random_indices = np.random.choice(len(train_combinations), size=min(len(train_combinations), 25000), replace=False)
        train_combinations = train_combinations[random_indices]

        X1_fno = np.zeros((len(train_combinations), 128, 7))
        y1_fno = np.zeros((len(train_combinations), 128))
        X2_fno = np.zeros((len(train_combinations), 128, 7))
        y2_fno = np.zeros((len(train_combinations), 128))

        for (i, combination) in enumerate(train_combinations):
            indices = np.where((d_full[:, 0] == combination[0]) & 
                            (d_full[:, 1] == combination[1]) &
                            (d_full[:, 2] == combination[2]) &
                            (d_full[:, 3] == combination[3]) &
                            (d_full[:, 4] == combination[4]) &
                            (d_full[:, 5] == combination[5]))
            indices = indices[0]

            phi_values = d_full[indices][:,6]
            
            X1_fno[i,:,:-1] = combination
            X1_fno[i,:,-1] = phi_values

            y1_fno[i,:] = d_full[indices][:,8]


            phi_values = d_full[indices][:,7]
            
            X2_fno[i,:,:-1] = combination
            X2_fno[i,:,-1] = phi_values

            y2_fno[i,:] = d_full[indices][:,9]

        print("Revised input size: {}; Revised output size: {}".format(X1_fno.shape, y1_fno.shape))
        device = 'cuda'
        X_train_gpu = torch.FloatTensor(X1_fno[:int(0.8*len(X1_fno))]).to(device)
        X_val_gpu = torch.FloatTensor(X1_fno[int(0.8*len(X1_fno)):]).to(device)

        y_train_gpu = torch.FloatTensor(y1_fno[:int(0.8*len(y1_fno))]).to(device)
        y_val_gpu = torch.FloatTensor(y1_fno[int(0.8*len(y1_fno)):]).to(device)

        # Train
        batch_size = 250000
        lr = 0.001

        epochs = 1000
        step_size = 50
        gamma = 0.5

        modes = 64
        width = 64
        fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
        fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "files/trained_models/fno/", "{}_C1_TENSION".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))

        print("")

        print("Revised input size: {}; Revised output size: {}".format(X2_fno.shape, y2_fno.shape))
        device = 'cuda'
        X_train_gpu = torch.FloatTensor(X2_fno[:int(0.8*len(X2_fno))]).to(device)
        X_val_gpu = torch.FloatTensor(X2_fno[int(0.8*len(X2_fno)):]).to(device)

        y_train_gpu = torch.FloatTensor(y2_fno[:int(0.8*len(y2_fno))]).to(device)
        y_val_gpu = torch.FloatTensor(y2_fno[int(0.8*len(y2_fno)):]).to(device)

        # Train
        batch_size = 250000
        lr = 0.001

        epochs = 1000
        step_size = 50
        gamma = 0.5

        modes = 64
        width = 64
        fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
        fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "files/trained_models/fno/", "{}_C2_TENSION".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))

        print("")

        print("---------------")
        print("BENDING Loading")
        print("---------------")
        
        ############################### RFR ###############################
        print("==========================================================")
        print("Training RFR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X1_train = d[:,:7]
        y1_train = d[:,10]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))

        rfr = RandomForestRegressor(max_depth=None)
        rfr.fit(X1_train, y1_train)
        # Saving trained model
        dump(rfr, 'files/trained_models/rfr/{}_C1_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,11]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        rfr = RandomForestRegressor(max_depth=None)
        rfr.fit(X2_train, y2_train)
        # Saving trained model
        dump(rfr, 'files/trained_models/rfr/{}_C2_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Training SVR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X1_train = d[:,:7]
        y1_train = d[:,10]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))

        svr = SVR()
        svr.fit(X1_train, y1_train)
        # Saving trained model
        dump(svr, 'files/trained_models/svr/{}_C1_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,11]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        svr = SVR()
        svr.fit(X2_train, y2_train)
        # Saving trained model
        dump(svr, 'files/trained_models/svr/{}_C2_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Training NN on a subset of size 500000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 500000), replace=False)
        d = d_full[random_indices]
        
        X1_train = d[:,:7]
        y1_train = d[:,10]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))
        
        device = 'cuda'
        EPOCHS = 150000
        # CPU -> GPU
        X_train_gpu = torch.FloatTensor(X1_train[:int(0.8*len(X1_train))]).to(device)
        X_val_gpu = torch.FloatTensor(X1_train[int(0.8*len(X1_train)):]).to(device)
        y_train_gpu = torch.FloatTensor(np.expand_dims(y1_train[:int(0.8*len(y1_train))], axis=-1)).to(device)
        y_val_gpu = torch.FloatTensor(np.expand_dims(y1_train[int(0.8*len(y1_train)):], axis=-1)).to(device)
        # Training
        net = custom_nn.Net10(X_train_gpu.shape[1], 15).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        FILENAME = "files/trained_models/nn/{}_C1_BENDING.pt".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        _, _ = custom_nn.train_model(net, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, loss_fn, optimizer, EPOCHS, FILENAME)
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,11]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        device = 'cuda'
        EPOCHS = 150000
        # CPU -> GPU
        X_train_gpu = torch.FloatTensor(X2_train[:int(0.8*len(X2_train))]).to(device)
        X_val_gpu = torch.FloatTensor(X2_train[int(0.8*len(X2_train)):]).to(device)
        y_train_gpu = torch.FloatTensor(np.expand_dims(y2_train[:int(0.8*len(y2_train))], axis=-1)).to(device)
        y_val_gpu = torch.FloatTensor(np.expand_dims(y2_train[int(0.8*len(y2_train)):], axis=-1)).to(device)
        # Training
        net = custom_nn.Net10(X_train_gpu.shape[1], 15).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        FILENAME = "files/trained_models/nn/{}_C2_BENDING.pt".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        _, _ = custom_nn.train_model(net, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, loss_fn, optimizer, EPOCHS, FILENAME)
        print("")
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Training FNO on a subset of size 25000 geometries (randomly sampled)")
        random_indices = np.random.choice(len(train_combinations), size=min(len(train_combinations), 25000), replace=False)
        train_combinations = train_combinations[random_indices]

        X1_fno = np.zeros((len(train_combinations), 128, 7))
        y1_fno = np.zeros((len(train_combinations), 128))
        X2_fno = np.zeros((len(train_combinations), 128, 7))
        y2_fno = np.zeros((len(train_combinations), 128))

        for (i, combination) in enumerate(train_combinations):
            indices = np.where((d_full[:, 0] == combination[0]) & 
                            (d_full[:, 1] == combination[1]) &
                            (d_full[:, 2] == combination[2]) &
                            (d_full[:, 3] == combination[3]) &
                            (d_full[:, 4] == combination[4]) &
                            (d_full[:, 5] == combination[5]))
            indices = indices[0]

            phi_values = d_full[indices][:,6]
            
            X1_fno[i,:,:-1] = combination
            X1_fno[i,:,-1] = phi_values

            y1_fno[i,:] = d_full[indices][:,10]


            phi_values = d_full[indices][:,7]
            
            X2_fno[i,:,:-1] = combination
            X2_fno[i,:,-1] = phi_values

            y2_fno[i,:] = d_full[indices][:,11]

        print("Revised input size: {}; Revised output size: {}".format(X1_fno.shape, y1_fno.shape))
        device = 'cuda'
        X_train_gpu = torch.FloatTensor(X1_fno[:int(0.8*len(X1_fno))]).to(device)
        X_val_gpu = torch.FloatTensor(X1_fno[int(0.8*len(X1_fno)):]).to(device)

        y_train_gpu = torch.FloatTensor(y1_fno[:int(0.8*len(y1_fno))]).to(device)
        y_val_gpu = torch.FloatTensor(y1_fno[int(0.8*len(y1_fno)):]).to(device)

        # Train
        batch_size = 250000
        lr = 0.001

        epochs = 1000
        step_size = 50
        gamma = 0.5

        modes = 64
        width = 64
        fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
        fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "files/trained_models/fno/", "{}_C1_BENDING".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))

        print("")

        print("Revised input size: {}; Revised output size: {}".format(X2_fno.shape, y2_fno.shape))
        device = 'cuda'
        X_train_gpu = torch.FloatTensor(X2_fno[:int(0.8*len(X2_fno))]).to(device)
        X_val_gpu = torch.FloatTensor(X2_fno[int(0.8*len(X2_fno)):]).to(device)

        y_train_gpu = torch.FloatTensor(y2_fno[:int(0.8*len(y2_fno))]).to(device)
        y_val_gpu = torch.FloatTensor(y2_fno[int(0.8*len(y2_fno)):]).to(device)

        # Train
        batch_size = 250000
        lr = 0.001

        epochs = 1000
        step_size = 50
        gamma = 0.5

        modes = 64
        width = 64
        fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
        fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "files/trained_models/fno/", "{}_C2_BENDING".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))

        print("")

        print("---------------")
        print("BEARING Loading")
        print("---------------")
        
        ############################### RFR ###############################
        print("==========================================================")
        print("Training RFR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X1_train = d[:,:7]
        y1_train = d[:,12]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))

        rfr = RandomForestRegressor(max_depth=None)
        rfr.fit(X1_train, y1_train)
        # Saving trained model
        dump(rfr, 'files/trained_models/rfr/{}_C1_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,13]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        rfr = RandomForestRegressor(max_depth=None)
        rfr.fit(X2_train, y2_train)
        # Saving trained model
        dump(rfr, 'files/trained_models/rfr/{}_C2_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Training SVR on a subset of size 100000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 100000), replace=False)
        d = d_full[random_indices]

        X1_train = d[:,:7]
        y1_train = d[:,12]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))

        svr = SVR()
        svr.fit(X1_train, y1_train)
        # Saving trained model
        dump(svr, 'files/trained_models/svr/{}_C1_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,13]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        svr = SVR()
        svr.fit(X2_train, y2_train)
        # Saving trained model
        dump(svr, 'files/trained_models/svr/{}_C2_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Training NN on a subset of size 500000 (randomly sampled)")
        random_indices = np.random.choice(len(d_full), size=min(len(d_full), 500000), replace=False)
        d = d_full[random_indices]
        
        X1_train = d[:,:7]
        y1_train = d[:,12]
        print("Revised input size: {}; Revised output size: {}".format(X1_train.shape, y1_train.shape))
        
        device = 'cuda'
        EPOCHS = 150000
        # CPU -> GPU
        X_train_gpu = torch.FloatTensor(X1_train[:int(0.8*len(X1_train))]).to(device)
        X_val_gpu = torch.FloatTensor(X1_train[int(0.8*len(X1_train)):]).to(device)
        y_train_gpu = torch.FloatTensor(np.expand_dims(y1_train[:int(0.8*len(y1_train))], axis=-1)).to(device)
        y_val_gpu = torch.FloatTensor(np.expand_dims(y1_train[int(0.8*len(y1_train)):], axis=-1)).to(device)
        # Training
        net = custom_nn.Net10(X_train_gpu.shape[1], 15).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        FILENAME = "files/trained_models/nn/{}_C1_BEARING.pt".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        _, _ = custom_nn.train_model(net, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, loss_fn, optimizer, EPOCHS, FILENAME)
        print("")

        X2_train = np.delete(d[:,:8], 6, axis=1)
        y2_train = d[:,13]
        print("Revised input size: {}; Revised output size: {}".format(X2_train.shape, y2_train.shape))

        device = 'cuda'
        EPOCHS = 150000
        # CPU -> GPU
        X_train_gpu = torch.FloatTensor(X2_train[:int(0.8*len(X2_train))]).to(device)
        X_val_gpu = torch.FloatTensor(X2_train[int(0.8*len(X2_train)):]).to(device)
        y_train_gpu = torch.FloatTensor(np.expand_dims(y2_train[:int(0.8*len(y2_train))], axis=-1)).to(device)
        y_val_gpu = torch.FloatTensor(np.expand_dims(y2_train[int(0.8*len(y2_train)):], axis=-1)).to(device)
        # Training
        net = custom_nn.Net10(X_train_gpu.shape[1], 15).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        FILENAME = "files/trained_models/nn/{}_C2_BEARING.pt".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        _, _ = custom_nn.train_model(net, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, loss_fn, optimizer, EPOCHS, FILENAME)
        print("")
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Training FNO on a subset of size 25000 geometries (randomly sampled)")
        random_indices = np.random.choice(len(train_combinations), size=min(len(train_combinations), 25000), replace=False)
        train_combinations = train_combinations[random_indices]

        X1_fno = np.zeros((len(train_combinations), 128, 7))
        y1_fno = np.zeros((len(train_combinations), 128))
        X2_fno = np.zeros((len(train_combinations), 128, 7))
        y2_fno = np.zeros((len(train_combinations), 128))

        for (i, combination) in enumerate(train_combinations):
            indices = np.where((d_full[:, 0] == combination[0]) & 
                            (d_full[:, 1] == combination[1]) &
                            (d_full[:, 2] == combination[2]) &
                            (d_full[:, 3] == combination[3]) &
                            (d_full[:, 4] == combination[4]) &
                            (d_full[:, 5] == combination[5]))
            indices = indices[0]

            phi_values = d_full[indices][:,6]
            
            X1_fno[i,:,:-1] = combination
            X1_fno[i,:,-1] = phi_values

            y1_fno[i,:] = d_full[indices][:,12]


            phi_values = d_full[indices][:,7]
            
            X2_fno[i,:,:-1] = combination
            X2_fno[i,:,-1] = phi_values

            y2_fno[i,:] = d_full[indices][:,13]

        print("Revised input size: {}; Revised output size: {}".format(X1_fno.shape, y1_fno.shape))
        device = 'cuda'
        X_train_gpu = torch.FloatTensor(X1_fno[:int(0.8*len(X1_fno))]).to(device)
        X_val_gpu = torch.FloatTensor(X1_fno[int(0.8*len(X1_fno)):]).to(device)

        y_train_gpu = torch.FloatTensor(y1_fno[:int(0.8*len(y1_fno))]).to(device)
        y_val_gpu = torch.FloatTensor(y1_fno[int(0.8*len(y1_fno)):]).to(device)

        # Train
        batch_size = 250000
        lr = 0.001

        epochs = 1000
        step_size = 50
        gamma = 0.5

        modes = 64
        width = 64
        fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
        fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "files/trained_models/fno/", "{}_C1_BEARING".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))

        print("")

        print("Revised input size: {}; Revised output size: {}".format(X2_fno.shape, y2_fno.shape))
        device = 'cuda'
        X_train_gpu = torch.FloatTensor(X2_fno[:int(0.8*len(X2_fno))]).to(device)
        X_val_gpu = torch.FloatTensor(X2_fno[int(0.8*len(X2_fno)):]).to(device)

        y_train_gpu = torch.FloatTensor(y2_fno[:int(0.8*len(y2_fno))]).to(device)
        y_val_gpu = torch.FloatTensor(y2_fno[int(0.8*len(y2_fno)):]).to(device)

        # Train
        batch_size = 250000
        lr = 0.001

        epochs = 1000
        step_size = 50
        gamma = 0.5

        modes = 64
        width = 64
        fno_model = FNO(len(phi_values), X_train_gpu.shape[-1], modes, width, device)
        fno_model.train_model(X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu, batch_size, lr, step_size, gamma, epochs, "files/trained_models/fno/", "{}_C2_BEARING".format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))

        print("")
    ##############################################################################
    ################################## Testing ###################################
    ##############################################################################
    elif what_to_do == "Test":
        print("{} on the quarter-ellipse corner crack dataset".format(what_to_do))

        # Loading dataset
        df_test = pd.read_csv("files/data/TWIN/CORNER_CRACK_COUNTERSUNK_HOLE/TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1_TEST.csv")
        df_test = df_test.drop(columns=['b/t'])
        df_test = df_test[(df_test["W/R"] >= 4) & (df_test["W/R"] <= 10)]
        df_test = df_test[(df_test["r/t"] >= 0.5) & (df_test["r/t"] <= 1.5)]
        test_combinations = df_test.iloc[:, 1:7].drop_duplicates().to_numpy()
        d = df_test.to_numpy()[:,1:]
        print("Dataset size: {}; Num geom: {}".format(d.shape, len(test_combinations)))

        print("---------------")
        print("Tension Loading")
        print("---------------")

        ############################### RFR ###############################
        print("==========================================================")
        print("Testing RFR")

        X1_test = d[::8,:7]
        y1_test = d[::8,8]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        rfr = load('files/trained_models/rfr/{}_C1_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = rfr.predict(X1_test)
        # Saving prediction
        np.save('files/predictions/rfr/{}_C1_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,9]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        rfr = load('files/trained_models/rfr/{}_C2_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = rfr.predict(X2_test)
        # Saving prediction
        np.save('files/predictions/rfr/{}_C2_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Testing SVR")
        
        X1_test = d[::8,:7]
        y1_test = d[::8,8]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        svr = load('files/trained_models/svr/{}_C1_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = svr.predict(X1_test)
        # Saving prediction
        np.save('files/predictions/svr/{}_C1_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,9]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        svr = load('files/trained_models/svr/{}_C2_TENSION.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = svr.predict(X2_test)
        # Saving prediction
        np.save('files/predictions/svr/{}_C2_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Testing NN")
        device = "cuda"
        X1_test = d[::8,:7]
        y1_test = d[::8,8]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        X_test_gpu = torch.FloatTensor(X1_test).to(device)
        net = custom_nn.Net10(X_test_gpu.shape[1], 15).to(device)
        FILENAME = 'files/trained_models/nn/{}_C1_TENSION.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        net.load_state_dict(torch.load(FILENAME, weights_only=False))
        with torch.no_grad():
            y_pred = net.forward(X_test_gpu).cpu().numpy()[:,0]

        # Saving predictions model
        np.save('files/predictions/nn/{}_C1_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,9]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        X_test_gpu = torch.FloatTensor(X2_test).to(device)
        net = custom_nn.Net10(X_test_gpu.shape[1], 15).to(device)
        FILENAME = 'files/trained_models/nn/{}_C2_TENSION.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        net.load_state_dict(torch.load(FILENAME, weights_only=False))
        with torch.no_grad():
            y_pred = net.forward(X_test_gpu).cpu().numpy()[:,0]

        # Saving predictions model
        np.save('files/predictions/nn/{}_C2_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Testing FNO")
        X1_fno = np.zeros((len(test_combinations), 128, 7))
        y1_fno = np.zeros((len(test_combinations), 128))
        X2_fno = np.zeros((len(test_combinations), 128, 7))
        y2_fno = np.zeros((len(test_combinations), 128))
        print("Input size: {}; Output size: {}".format(X1_fno.shape, y1_fno.shape))

        for (i, combination) in enumerate(test_combinations):
            indices = np.where((d[:, 0] == combination[0]) & 
                            (d[:, 1] == combination[1]) &
                            (d[:, 2] == combination[2]) &
                            (d[:, 3] == combination[3]) &
                            (d[:, 4] == combination[4]) &
                            (d[:, 5] == combination[5])) 
            indices = indices[0]

            phi_values = d[indices][:,6]
            
            X1_fno[i,:,:-1] = combination
            X1_fno[i,:,-1] = phi_values

            y1_fno[i,:] = d[indices][:,8]


            phi_values = d[indices][:,7]
            
            X2_fno[i,:,:-1] = combination
            X2_fno[i,:,-1] = phi_values

            y2_fno[i,:] = d[indices][:,9]

        X_test_gpu = torch.FloatTensor(X1_fno).to(device)
        y_test_gpu = torch.FloatTensor(y1_fno).to(device)

        len_phi_values = 128
        modes = 64
        width = 64
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C1_TENSION.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C1_TENSION.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('files/predictions/fno/{}_C1_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)

        print("")

        X_test_gpu = torch.FloatTensor(X2_fno).to(device)
        y_test_gpu = torch.FloatTensor(y2_fno).to(device)

        len_phi_values = 128
        modes = 64
        width = 64
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C2_TENSION.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C2_TENSION.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('files/predictions/fno/{}_C2_TENSION.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)

        print("")

        print("---------------")
        print("BENDING Loading")
        print("---------------")

        ############################### RFR ###############################
        print("==========================================================")
        print("Testing RFR")

        X1_test = d[::8,:7]
        y1_test = d[::8,10]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        rfr = load('files/trained_models/rfr/{}_C1_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = rfr.predict(X1_test)
        # Saving prediction
        np.save('files/predictions/rfr/{}_C1_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,11]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        rfr = load('files/trained_models/rfr/{}_C2_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = rfr.predict(X2_test)
        # Saving prediction
        np.save('files/predictions/rfr/{}_C2_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Testing SVR")
        
        X1_test = d[::8,:7]
        y1_test = d[::8,10]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        svr = load('files/trained_models/svr/{}_C1_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = svr.predict(X1_test)
        # Saving prediction
        np.save('files/predictions/svr/{}_C1_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,11]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        svr = load('files/trained_models/svr/{}_C2_BENDING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = svr.predict(X2_test)
        # Saving prediction
        np.save('files/predictions/svr/{}_C2_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Testing NN")
        device = "cuda"
        X1_test = d[::8,:7]
        y1_test = d[::8,10]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        X_test_gpu = torch.FloatTensor(X1_test).to(device)
        net = custom_nn.Net10(X_test_gpu.shape[1], 15).to(device)
        FILENAME = 'files/trained_models/nn/{}_C1_BENDING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        net.load_state_dict(torch.load(FILENAME, weights_only=False))
        with torch.no_grad():
            y_pred = net.forward(X_test_gpu).cpu().numpy()[:,0]

        # Saving predictions model
        np.save('files/predictions/nn/{}_C1_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,11]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        X_test_gpu = torch.FloatTensor(X2_test).to(device)
        net = custom_nn.Net10(X_test_gpu.shape[1], 15).to(device)
        FILENAME = 'files/trained_models/nn/{}_C2_BENDING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        net.load_state_dict(torch.load(FILENAME, weights_only=False))
        with torch.no_grad():
            y_pred = net.forward(X_test_gpu).cpu().numpy()[:,0]

        # Saving predictions model
        np.save('files/predictions/nn/{}_C2_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Testing FNO")
        X1_fno = np.zeros((len(test_combinations), 128, 7))
        y1_fno = np.zeros((len(test_combinations), 128))
        X2_fno = np.zeros((len(test_combinations), 128, 7))
        y2_fno = np.zeros((len(test_combinations), 128))
        print("Input size: {}; Output size: {}".format(X1_fno.shape, y1_fno.shape))

        for (i, combination) in enumerate(test_combinations):
            indices = np.where((d[:, 0] == combination[0]) & 
                            (d[:, 1] == combination[1]) &
                            (d[:, 2] == combination[2]) &
                            (d[:, 3] == combination[3]) &
                            (d[:, 4] == combination[4]) &
                            (d[:, 5] == combination[5])) 
            indices = indices[0]

            phi_values = d[indices][:,6]
            
            X1_fno[i,:,:-1] = combination
            X1_fno[i,:,-1] = phi_values

            y1_fno[i,:] = d[indices][:,10]


            phi_values = d[indices][:,7]
            
            X2_fno[i,:,:-1] = combination
            X2_fno[i,:,-1] = phi_values

            y2_fno[i,:] = d[indices][:,11]

        X_test_gpu = torch.FloatTensor(X1_fno).to(device)
        y_test_gpu = torch.FloatTensor(y1_fno).to(device)

        len_phi_values = 128
        modes = 64
        width = 64
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C1_BENDING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C1_BENDING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('files/predictions/fno/{}_C1_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)

        print("")

        X_test_gpu = torch.FloatTensor(X2_fno).to(device)
        y_test_gpu = torch.FloatTensor(y2_fno).to(device)

        len_phi_values = 128
        modes = 64
        width = 64
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C2_BENDING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C2_BENDING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('files/predictions/fno/{}_C2_BENDING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)

        print("")

        print("---------------")
        print("BEARING Loading")
        print("---------------")

        ############################### RFR ###############################
        print("==========================================================")
        print("Testing RFR")

        X1_test = d[::8,:7]
        y1_test = d[::8,12]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        rfr = load('files/trained_models/rfr/{}_C1_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = rfr.predict(X1_test)
        # Saving prediction
        np.save('files/predictions/rfr/{}_C1_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,13]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        rfr = load('files/trained_models/rfr/{}_C2_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = rfr.predict(X2_test)
        # Saving prediction
        np.save('files/predictions/rfr/{}_C2_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        ############################### SVR ###############################
        print("==========================================================")
        print("Testing SVR")
        
        X1_test = d[::8,:7]
        y1_test = d[::8,12]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        svr = load('files/trained_models/svr/{}_C1_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = svr.predict(X1_test)
        # Saving prediction
        np.save('files/predictions/svr/{}_C1_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,13]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        svr = load('files/trained_models/svr/{}_C2_BEARING.joblib'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"))
        y_pred = svr.predict(X2_test)
        # Saving prediction
        np.save('files/predictions/svr/{}_C2_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        ############################### NN ###############################
        print("==========================================================")
        print("Testing NN")
        device = "cuda"
        X1_test = d[::8,:7]
        y1_test = d[::8,12]
        print("Input size: {}; Output size: {}".format(X1_test.shape, y1_test.shape))

        X_test_gpu = torch.FloatTensor(X1_test).to(device)
        net = custom_nn.Net10(X_test_gpu.shape[1], 15).to(device)
        FILENAME = 'files/trained_models/nn/{}_C1_BEARING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        net.load_state_dict(torch.load(FILENAME, weights_only=False))
        with torch.no_grad():
            y_pred = net.forward(X_test_gpu).cpu().numpy()[:,0]

        # Saving predictions model
        np.save('files/predictions/nn/{}_C1_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")

        X2_test = np.delete(d[::8,:8], 6, axis=1)
        y2_test = d[::8,13]
        print("Input size: {}; Output size: {}".format(X2_test.shape, y2_test.shape))

        X_test_gpu = torch.FloatTensor(X2_test).to(device)
        net = custom_nn.Net10(X_test_gpu.shape[1], 15).to(device)
        FILENAME = 'files/trained_models/nn/{}_C2_BEARING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1")
        net.load_state_dict(torch.load(FILENAME, weights_only=False))
        with torch.no_grad():
            y_pred = net.forward(X_test_gpu).cpu().numpy()[:,0]

        # Saving predictions model
        np.save('files/predictions/nn/{}_C2_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)
        print("")
        
        ############################### FNO ###############################
        print("==========================================================")
        print("Testing FNO")
        X1_fno = np.zeros((len(test_combinations), 128, 7))
        y1_fno = np.zeros((len(test_combinations), 128))
        X2_fno = np.zeros((len(test_combinations), 128, 7))
        y2_fno = np.zeros((len(test_combinations), 128))
        print("Input size: {}; Output size: {}".format(X1_fno.shape, y1_fno.shape))

        for (i, combination) in enumerate(test_combinations):
            indices = np.where((d[:, 0] == combination[0]) & 
                            (d[:, 1] == combination[1]) &
                            (d[:, 2] == combination[2]) &
                            (d[:, 3] == combination[3]) &
                            (d[:, 4] == combination[4]) &
                            (d[:, 5] == combination[5])) 
            indices = indices[0]

            phi_values = d[indices][:,6]
            
            X1_fno[i,:,:-1] = combination
            X1_fno[i,:,-1] = phi_values

            y1_fno[i,:] = d[indices][:,12]


            phi_values = d[indices][:,7]
            
            X2_fno[i,:,:-1] = combination
            X2_fno[i,:,-1] = phi_values

            y2_fno[i,:] = d[indices][:,13]

        X_test_gpu = torch.FloatTensor(X1_fno).to(device)
        y_test_gpu = torch.FloatTensor(y1_fno).to(device)

        len_phi_values = 128
        modes = 64
        width = 64
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C1_BEARING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C1_BEARING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('files/predictions/fno/{}_C1_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)

        print("")

        X_test_gpu = torch.FloatTensor(X2_fno).to(device)
        y_test_gpu = torch.FloatTensor(y2_fno).to(device)

        len_phi_values = 128
        modes = 64
        width = 64
        fno_loaded_model = FNO1d(len_phi_values, X_test_gpu.shape[-1], modes, width).to(device)
        if device == "cpu":
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C2_BEARING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), map_location=torch.device('cpu'), weights_only=False))
        else:
            fno_loaded_model.load_state_dict(torch.load('files/trained_models/fno/fno_{}_C2_BEARING.pt'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"),weights_only=False))
        with torch.no_grad():
            y_pred = fno_loaded_model(X_test_gpu).cpu().numpy()[:,:,0]

        # Saving predictions model
        np.save('files/predictions/fno/{}_C2_BEARING.npy'.format("TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1"), y_pred)

        print("")
         
    ##############################################################################
    ################################## Invalid ###################################
    ##############################################################################
    else:
        raise ValueError("Inputs can only be Train or Test")