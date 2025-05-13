import numpy as np
import pandas as pd

def mean_normalized_l2(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true -  y_pred, ord=2, axis=1) / np.linalg.norm(y_true, ord=2, axis=1))

def mean_normalized_abs(y_true, y_pred):
    return np.mean(np.abs((y_true -  y_pred)/y_true))

if __name__ == "__main__":
    ###########################################################################
    ############################## SINGLE CRACK ###############################
    ###########################################################################
    SINGLE_CRACK_DATASET_LOCS = ['files/data/SINGLE_CRACK/SURFACE_CRACK/SURFACE_CRACK',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_STRAIGHT_HOLE/CORNER_CRACK_BH_QUARTER_ELLIPSE',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_STRAIGHT_HOLE/CORNER_CRACK_BH_THROUGH_THICKNESS',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_COUNTERSUNK_HOLE/CORNER_CRACK_CS1_QUARTER_ELLIPSE',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_COUNTERSUNK_HOLE/CORNER_CRACK_CS2_QUARTER_ELLIPSE',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_COUNTERSUNK_HOLE/CORNER_CRACK_CS2_THROUGH_CS_THICKNESS',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_COUNTERSUNK_HOLE/CORNER_CRACK_CS3_QUARTER_ELLIPSE',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_COUNTERSUNK_HOLE/CORNER_CRACK_CS4_QUARTER_ELLIPSE',
                                 'files/data/SINGLE_CRACK/CORNER_CRACK_COUNTERSUNK_HOLE/CORNER_CRACK_CS4_THROUGH_CS_THICKNESS']
    
    LOADINGS = ['TENSION', 'BENDING', 'BEARING']

    single_l2 = np.zeros((len(SINGLE_CRACK_DATASET_LOCS), 3, 4))
    single_abs = np.zeros((len(SINGLE_CRACK_DATASET_LOCS), 3, 4))

    for (i, data_name) in enumerate(SINGLE_CRACK_DATASET_LOCS):
        # Extract the last part of the path
        crack_type = data_name.split('/')[-1]
        
        if crack_type == "SURFACE_CRACK":
            df_test = pd.read_csv(data_name+'_TEST.csv')
            test_combinations = df_test.iloc[:, 1:4].drop_duplicates().to_numpy()
            d = df_test.to_numpy()[:,1:]

            # RFR
            y_test = d[:,-1].reshape(-1, 128)
            y_pred = np.load('files/predictions/rfr/{}_TENSION.npy'.format(crack_type)).reshape(-1, 128)
            single_l2[i,0,0] = mean_normalized_l2(y_test, y_pred)
            single_abs[i,0,0] = mean_normalized_abs(y_test, y_pred)

            # SVR
            y_test = d[:,-1].reshape(-1, 128)
            y_pred = np.load('files/predictions/svr/{}_TENSION.npy'.format(crack_type)).reshape(-1, 128)
            single_l2[i,0,1] = mean_normalized_l2(y_test, y_pred)
            single_abs[i,0,1] = mean_normalized_abs(y_test, y_pred)

            # NN
            y_test = d[:,-1].reshape(-1, 128)
            y_pred = np.load('files/predictions/nn/{}_TENSION.npy'.format(crack_type)).reshape(-1, 128)
            single_l2[i,0,2] = mean_normalized_l2(y_test, y_pred)
            single_abs[i,0,2] = mean_normalized_abs(y_test, y_pred)

            # FNO
            y_test = d[:,-1].reshape(-1, 128)
            y_pred = np.load('files/predictions/fno/{}_TENSION.npy'.format(crack_type))
            single_l2[i,0,3] = mean_normalized_l2(y_test, y_pred)
            single_abs[i,0,3] = mean_normalized_abs(y_test, y_pred)
        
        elif crack_type == "CORNER_CRACK_BH_THROUGH_THICKNESS":
            df_test = pd.read_csv(data_name+'_TEST.csv')
            df_test = df_test.drop(columns=['b/t'])
            df_test = df_test[(df_test["r/t"] >= 0.5) & (df_test["r/t"] <= 1.5)]
            test_combinations = df_test.iloc[:, 1:5].drop_duplicates().to_numpy()
            d = df_test.to_numpy()[:,1:]

            for (j, LOADING) in enumerate(LOADINGS):
                # RFR
                y_test = d[::8,-3+j].reshape(-1, 16)
                y_pred = np.load('files/predictions/rfr/{}_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                single_l2[i,j,0] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,0] = mean_normalized_abs(y_test, y_pred)

                # SVR
                y_test = d[::8,-3+j].reshape(-1, 16)
                y_pred = np.load('files/predictions/svr/{}_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                single_l2[i,j,1] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,1] = mean_normalized_abs(y_test, y_pred)

                # NN
                y_test = d[::8,-3+j].reshape(-1, 16)
                y_pred = np.load('files/predictions/nn/{}_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                single_l2[i,j,2] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,2] = mean_normalized_abs(y_test, y_pred)

                # FNO
                y_test = d[:,-3+j].reshape(-1, 128)
                y_pred = np.load('files/predictions/fno/{}_{}.npy'.format(crack_type, LOADING))
                single_l2[i,j,3] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,3] = mean_normalized_abs(y_test, y_pred)

        else:
            df_test = pd.read_csv(data_name+'_TEST.csv')
            df_test = df_test.drop(columns=['b/t'])
            df_test = df_test[(df_test["W/R"] >= 2) & (df_test["W/R"] <= 20)]
            df_test = df_test[(df_test["r/t"] >= 0.5) & (df_test["r/t"] <= 1.5)]
            test_combinations = df_test.iloc[:, 1:5].drop_duplicates().to_numpy()
            d = df_test.to_numpy()[:,1:]

            for (j, LOADING) in enumerate(LOADINGS):
                # RFR
                y_test = d[::8,-3+j].reshape(-1, 16)
                y_pred = np.load('files/predictions/rfr/{}_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                single_l2[i,j,0] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,0] = mean_normalized_abs(y_test, y_pred)

                # SVR
                y_test = d[::8,-3+j].reshape(-1, 16)
                y_pred = np.load('files/predictions/svr/{}_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                single_l2[i,j,1] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,1] = mean_normalized_abs(y_test, y_pred)

                # NN
                y_test = d[::8,-3+j].reshape(-1, 16)
                y_pred = np.load('files/predictions/nn/{}_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                single_l2[i,j,2] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,2] = mean_normalized_abs(y_test, y_pred)

                # FNO
                y_test = d[:,-3+j].reshape(-1, 128)
                y_pred = np.load('files/predictions/fno/{}_{}.npy'.format(crack_type, LOADING))
                single_l2[i,j,3] = mean_normalized_l2(y_test, y_pred)
                single_abs[i,j,3] = mean_normalized_abs(y_test, y_pred)

    print("----------------------------------------------------------")
    print("----------------- SINGLE CRACK L2 ERROR ------------------")
    print("----------------------------------------------------------")
    for (i, data_name) in enumerate(SINGLE_CRACK_DATASET_LOCS):
        # Extract the last part of the path
        crack_type = data_name.split('/')[-1]
        if crack_type == "SURFACE_CRACK":
            print("{} (TENSION) | {:.4f} & {:.4f} & {:.4f} & {:.4f}".format(crack_type, 
                                                  single_l2[i,0,0],
                                                  single_l2[i,0,1],
                                                  single_l2[i,0,2],
                                                  single_l2[i,0,3]))
            print("----------------------------------------------------------")
        else:
            for (j, LOADING) in enumerate(LOADINGS):
                print("{} ({}) | {:.4f} & {:.4f} & {:.4f} & {:.4f}".format(crack_type, LOADING,
                                                      single_l2[i,j,0],
                                                      single_l2[i,j,1],
                                                      single_l2[i,j,2],
                                                      single_l2[i,j,3]))
                
                if LOADING == "BEARING":
                    print("----------------------------------------------------------")


    print("")

    print("----------------------------------------------------------")
    print("---------------- SINGLE CRACK ABS ERROR ------------------")
    print("----------------------------------------------------------")
    for (i, data_name) in enumerate(SINGLE_CRACK_DATASET_LOCS):
        # Extract the last part of the path
        crack_type = data_name.split('/')[-1]
        if crack_type == "SURFACE_CRACK":
            print("{} (TENSION) | {:.4f} & {:.4f} & {:.4f} & {:.4f}".format(crack_type, 
                                                  single_abs[i,0,0],
                                                  single_abs[i,0,1],
                                                  single_abs[i,0,2],
                                                  single_abs[i,0,3]))
            print("----------------------------------------------------------")
        else:
            for (j, LOADING) in enumerate(LOADINGS):
                print("{} ({}) | {:.4f} & {:.4f} & {:.4f} & {:.4f}".format(crack_type, LOADING,
                                                      single_abs[i,j,0],
                                                      single_abs[i,j,1],
                                                      single_abs[i,j,2],
                                                      single_abs[i,j,3]))
                
                if LOADING == "BEARING":
                    print("----------------------------------------------------------")

    print("")
    ###########################################################################
    ############################### TWIN CRACK ################################
    ###########################################################################
    TWIN_CRACK_DATASET_LOCS = ['files/data/TWIN/CORNER_CRACK_STRAIGHT_HOLE/TWIN_CORNER_CRACK_BH_QUARTER_ELLIPSE',
                               'files/data/TWIN/CORNER_CRACK_STRAIGHT_HOLE/TWIN_CORNER_CRACK_BH_THROUGH_THICKNESS',
                               'files/data/TWIN/CORNER_CRACK_COUNTERSUNK_HOLE/TWIN_CORNER_CRACK_CS2_QUARTER_ELLIPSE_PART_1',
                               'files/data/TWIN/CORNER_CRACK_COUNTERSUNK_HOLE/TWIN_CORNER_CRACK_CS2_THROUGH_THICKNESS']
    
    LOADINGS = ['TENSION', 'BENDING', 'BEARING']

    TWIN_l2 = np.zeros((len(TWIN_CRACK_DATASET_LOCS), 3, 4, 2))
    TWIN_abs = np.zeros((len(TWIN_CRACK_DATASET_LOCS), 3, 4, 2))

    for (i, data_name) in enumerate(TWIN_CRACK_DATASET_LOCS):
        for (j, LOADING) in enumerate(LOADINGS):
            # Extract the last part of the path
            crack_type = data_name.split('/')[-1]
            
            if crack_type == "TWIN_CORNER_CRACK_BH_QUARTER_ELLIPSE" or crack_type == "TWIN_CORNER_CRACK_BH_THROUGH_THICKNESS":
                df_test = pd.read_csv(data_name+'_TEST.csv')
                df_test = df_test.drop(columns=['b/t'])
                df_test = df_test[(df_test["r/t"] >= 0.5) & (df_test["r/t"] <= 1.5)]
                test_combinations = df_test.iloc[:, 1:7].drop_duplicates().to_numpy()
                d = df_test.to_numpy()[:,1:]

                # RFR
                y1_test = d[::8,8 + j*2].reshape(-1, 16)
                y1_pred = np.load('files/predictions/rfr/{}_C1_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,0,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,0,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[::8,8 + j*2 + 1].reshape(-1, 16)
                y2_pred = np.load('files/predictions/rfr/{}_C2_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,0,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,0,1] = mean_normalized_abs(y2_test, y2_pred)

                # SVR
                y1_test = d[::8,8 + j*2].reshape(-1, 16)
                y1_pred = np.load('files/predictions/svr/{}_C1_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,1,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,1,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[::8,8 + j*2 + 1].reshape(-1, 16)
                y2_pred = np.load('files/predictions/svr/{}_C2_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,1,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,1,1] = mean_normalized_abs(y2_test, y2_pred)

                # NN
                y1_test = d[::8,8 + j*2].reshape(-1, 16)
                y1_pred = np.load('files/predictions/nn/{}_C1_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,2,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,2,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[::8,8 + j*2 + 1].reshape(-1, 16)
                y2_pred = np.load('files/predictions/nn/{}_C2_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,2,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,2,1] = mean_normalized_abs(y2_test, y2_pred)

                # FNO
                y1_test = d[:,8 + j*2].reshape(-1, 128)
                y1_pred = np.load('files/predictions/fno/{}_C1_{}.npy'.format(crack_type, LOADING))
                TWIN_l2[i,j,3,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,3,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[:,8 + j*2 + 1].reshape(-1, 128)
                y2_pred = np.load('files/predictions/fno/{}_C2_{}.npy'.format(crack_type, LOADING))
                TWIN_l2[i,j,3,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,3,1] = mean_normalized_abs(y2_test, y2_pred)
            
            else:
                df_test = pd.read_csv(data_name+'_TEST.csv')
                df_test = df_test.drop(columns=['b/t'])
                df_test = df_test[(df_test["W/R"] >= 4) & (df_test["W/R"] <= 10)]
                df_test = df_test[(df_test["r/t"] >= 0.5) & (df_test["r/t"] <= 1.5)]
                test_combinations = df_test.iloc[:, 1:7].drop_duplicates().to_numpy()
                d = df_test.to_numpy()[:,1:]

                # RFR
                y1_test = d[::8,8 + j*2].reshape(-1, 16)
                y1_pred = np.load('files/predictions/rfr/{}_C1_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,0,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,0,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[::8,8 + j*2 + 1].reshape(-1, 16)
                y2_pred = np.load('files/predictions/rfr/{}_C2_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,0,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,0,1] = mean_normalized_abs(y2_test, y2_pred)

                # SVR
                y1_test = d[::8,8 + j*2].reshape(-1, 16)
                y1_pred = np.load('files/predictions/svr/{}_C1_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,1,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,1,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[::8,8 + j*2 + 1].reshape(-1, 16)
                y2_pred = np.load('files/predictions/svr/{}_C2_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,1,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,1,1] = mean_normalized_abs(y2_test, y2_pred)

                # NN
                y1_test = d[::8,8 + j*2].reshape(-1, 16)
                y1_pred = np.load('files/predictions/nn/{}_C1_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,2,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,2,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[::8,8 + j*2 + 1].reshape(-1, 16)
                y2_pred = np.load('files/predictions/nn/{}_C2_{}.npy'.format(crack_type, LOADING)).reshape(-1, 16)
                TWIN_l2[i,j,2,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,2,1] = mean_normalized_abs(y2_test, y2_pred)

                # FNO
                y1_test = d[:,8 + j*2].reshape(-1, 128)
                y1_pred = np.load('files/predictions/fno/{}_C1_{}.npy'.format(crack_type, LOADING))
                TWIN_l2[i,j,3,0] = mean_normalized_l2(y1_test, y1_pred)
                TWIN_abs[i,j,3,0] = mean_normalized_abs(y1_test, y1_pred)

                y2_test = d[:,8 + j*2 + 1].reshape(-1, 128)
                y2_pred = np.load('files/predictions/fno/{}_C2_{}.npy'.format(crack_type, LOADING))
                TWIN_l2[i,j,3,1] = mean_normalized_l2(y2_test, y2_pred)
                TWIN_abs[i,j,3,1] = mean_normalized_abs(y2_test, y2_pred)

    print("----------------------------------------------------------")
    print("----------------- TWIN CRACK L2 ERROR ------------------")
    print("----------------------------------------------------------")
    for (i, data_name) in enumerate(TWIN_CRACK_DATASET_LOCS):
        # Extract the last part of the path
        crack_type = data_name.split('/')[-1]
        for (j, LOADING) in enumerate(LOADINGS):
            print("{} ({}) | {:.3f} & {:.3f}   & {:.3f} & {:.3f}   & {:.3f} & {:.3f}   & {:.3f} & {:.3f}".format(crack_type, LOADING,
                                                                                 TWIN_l2[i,j,0,0], TWIN_l2[i,j,0,1],
                                                                                 TWIN_l2[i,j,1,0], TWIN_l2[i,j,1,1],
                                                                                 TWIN_l2[i,j,2,0], TWIN_l2[i,j,2,1],
                                                                                 TWIN_l2[i,j,3,0], TWIN_l2[i,j,3,1]))
            
            if LOADING == "BEARING":
                print("----------------------------------------------------------")


    print("")

    print("----------------------------------------------------------")
    print("---------------- TWIN CRACK ABS ERROR ------------------")
    print("----------------------------------------------------------")
    for (i, data_name) in enumerate(TWIN_CRACK_DATASET_LOCS):
        # Extract the last part of the path
        crack_type = data_name.split('/')[-1]
        for (j, LOADING) in enumerate(LOADINGS):
            print("{} ({}) | {:.3f} & {:.3f}   & {:.3f} & {:.3f}   & {:.3f} & {:.3f}   & {:.3f} & {:.3f}".format(crack_type, LOADING,
                                                                                 TWIN_abs[i,j,0,0], TWIN_abs[i,j,0,1],
                                                                                 TWIN_abs[i,j,1,0], TWIN_abs[i,j,1,1],
                                                                                 TWIN_abs[i,j,2,0], TWIN_abs[i,j,2,1],
                                                                                 TWIN_abs[i,j,3,0], TWIN_abs[i,j,3,1]))
            
            if LOADING == "BEARING":
                print("----------------------------------------------------------")