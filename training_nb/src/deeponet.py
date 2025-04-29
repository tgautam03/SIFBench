import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import itertools

class Net(nn.Module):
    def __init__(self, layers, act=nn.Tanh()):
        super(Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
        
    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x
    
class DeepONet:
    def __init__(self, branch_layers, trunk_layers, K, device, testing=False, branch_net_loc=None, trunk_net_loc=None):
        self.device = device
        
        # Branch Net
        self.branch_net = Net(branch_layers + [K], act=nn.Tanh()).to(self.device)
        if testing:
            if device == "cpu":
                self.branch_net.load_state_dict(torch.load(branch_net_loc, map_location=torch.device('cpu'), weights_only=False))
            else:
                self.branch_net.load_state_dict(torch.load(branch_net_loc, weights_only=False))

        # Trunk Net
        self.trunk_net = Net(trunk_layers + [K], act=nn.Tanh()).to(self.device)
        if testing:
            if device == "cpu":
                self.trunk_net.load_state_dict(torch.load(trunk_net_loc, map_location=torch.device('cpu'), weights_only=False))
            else:
                self.trunk_net.load_state_dict(torch.load(trunk_net_loc, weights_only=False))

        # Metrics 
        self.best_nrmse = None
        self.best_epoch = None
        self.train_nrmse = []
        self.test_nrmse = []
        self.train_nmse = []
        self.test_nmse = []

        # Best prediction
        self.y_pred_test_best = None

    def forward(self, branch_in, trunk_in):
        branch_pred = self.branch_net(branch_in) # N x K
        trunk_pred = self.trunk_net(trunk_in) # D^2 x K
        y = torch.matmul(branch_pred, trunk_pred.transpose(0,1))
        return y

    def train(self, branch_in_train, trunk_in_train, y_train, branch_in_test, trunk_in_test, y_test, lr, epochs, save_loc, save_name):
        params = list(itertools.chain(self.branch_net.parameters(), self.trunk_net.parameters()))
        minimizer = Adam(params, lr=lr)
        for epoch in range(epochs):
            minimizer.zero_grad()
            
            # Training pass
            y_pred_train = self.forward(branch_in_train, trunk_in_train)
            loss = (y_pred_train.reshape(y_train.shape) - y_train).square().mean()
            
            # Training/Testing eval
            with torch.no_grad():
                # Test preds
                y_pred_test = self.forward(branch_in_test, trunk_in_test)
                self._callback(y_pred_train, y_train, y_pred_test, y_test, epoch, save_loc, save_name)

            # Training update
            loss.backward(retain_graph=True)
            minimizer.step()
            
            # Printing
            if epoch % 1000 == 0:
                print("Epoch: {}; train loss: {}; train nrmse: {}; train nmse: {}; test nrmse: {}; test nmse: {}; best test epoch: {}; best test nrmse: {}".format(epoch, loss.item(), self.train_nrmse[-1], self.train_nmse[-1],
                                                                                                                                                                    self.test_nrmse[-1], self.test_nmse[-1],
                                                                                                                                                                    self.best_epoch, self.best_nrmse))

    def _callback(self, y_pred_train, y_train, y_pred_test, y_test, epoch, save_loc, save_name):
        # NRMSE
        tr_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_train - y_train)))/torch.sqrt(torch.square(y_train).mean())
        te_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_test - y_test)))/torch.sqrt(torch.square(y_test).mean())

        # NMSE
        tr_nmse = torch.mean(torch.square((y_pred_train - y_train)/y_train))
        te_nmse = torch.mean(torch.square((y_pred_test - y_test)/y_test))

        # Saving
        self.train_nrmse.append(tr_nrmse)
        self.test_nrmse.append(te_nrmse)
        self.train_nmse.append(tr_nmse)
        self.test_nmse.append(te_nmse)

        if self.best_nrmse is None or self.best_nrmse>te_nrmse:
            self.best_nrmse = te_nrmse
            self.best_epoch = epoch
            self.y_pred_test_best = y_pred_test
            torch.save(self.branch_net.state_dict(), save_loc + "branch_{}.pt".format(save_name))
            torch.save(self.trunk_net.state_dict(), save_loc + "trunk_{}.pt".format(save_name))
        
        return 