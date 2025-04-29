import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from timeit import default_timer
from tqdm import tqdm

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, num_phi_samples, d_in, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(d_in, self.width) # input channel is d_in: (feat_1, feat_2, .. feat_n , phi)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, num_phi_samples)
        self.fc2 = nn.Linear(num_phi_samples, 1)

    def forward(self, x):
        # x = self.fc_(x)
        # x = torch.unsqueeze(x, -1)
        
        # x: N x num_phi_samples x d_in
        x = self.fc0(x)     # N x num_phi_samples x D
        x = x.permute(0, 2, 1)  # N X D X num_phi_samples

        x1 = self.conv0(x)  
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x.permute(0, 2, 1)
        
        x = self.fc1(x)
        
        x = F.gelu(x)
        x = self.fc2(x)

        return x
    

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
class FNO:
    def __init__(self, num_phi_samples, d_in, modes, width, device):
        self.device = device

        # NN Model
        self.model = FNO1d(num_phi_samples, d_in, modes, width).to(self.device)

        # Metrics 
        self.best_nrmse = None
        self.best_epoch = None
        self.train_nrmse = []
        self.test_nrmse = []
        self.train_nmse = []
        self.test_nmse = []

        # Best prediction
        self.y_pred_test_best = None

    def train(self, X_train, y_train, X_test, y_test, batch_size, lr, step_size, gamma, epochs, save_loc, save_name):
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        myloss = LpLoss(size_average=False)

        for epoch in tqdm(range(epochs)):
            self.model.train()

            # Training passes
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = self.model(x)

                loss = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
                loss.backward() # use the l2 relative loss

                optimizer.step()
                train_loss += loss.item()
            train_loss /= X_train.shape[0]
            scheduler.step()

            # Training/Testing eval
            with torch.no_grad():
                # Train preds
                y_pred_train = self.model(X_train.to(self.device))

                # Test preds
                y_pred_test = self.model(X_test.to(self.device))

                self._callback(y_pred_train[:,:,0], y_train, y_pred_test[:,:,0], y_test, epoch, save_loc, save_name)

            # # Printing
            # if epoch % 10 == 0:
            #     print("Epoch: {}; train loss: {}; train nrmse: {}; train nmse: {}; test nrmse: {}; test nmse: {}; best test epoch: {}; best test nrmse: {}".format(epoch, train_loss, self.train_nrmse[-1], self.train_nmse[-1],
            #                                                                                                                                                         self.test_nrmse[-1], self.test_nmse[-1],
            #                                                                                                                                                         self.best_epoch, self.best_nrmse))
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
            torch.save(self.model.state_dict(), save_loc + "fno_{}.pt".format(save_name))
        
        return 