import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from timeit import default_timer
from tqdm import tqdm

################################################################
#  1d fourier layer
################################################################
# Provided LpLoss class
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
                # Added small epsilon to avoid division by zero
                epsilon = 1e-8
                return torch.mean(diff_norms / (y_norms + epsilon))
            else:
                 # Added small epsilon to avoid division by zero
                epsilon = 1e-8
                return torch.sum(diff_norms / (y_norms + epsilon))


        # Added small epsilon to avoid division by zero
        epsilon = 1e-8
        return diff_norms / (y_norms + epsilon)

    def __call__(self, x, y):
        return self.rel(x, y)

# Provided SpectralConv1d class
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

# Provided FNO1d class
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


# Modified FNO class inheriting from nn.Module and adding tqdm postfix updates
class FNO(torch.nn.Module): # <--- Added inheritance from torch.nn.Module
    def __init__(self, num_phi_samples, d_in, modes, width, device):
        super(FNO, self).__init__() # <--- Added call to the parent class constructor
        self.device = device

        # NN Model (assuming FNO1d is an nn.Module)
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

    # Added a forward method as is standard for nn.Module
    def forward(self, x):
         return self.model(x)


    def train_model(self, X_train, y_train, X_test, y_test, batch_size, lr, step_size, gamma, epochs, save_loc, save_name):
        # Ensure data is on the correct device before creating DataLoader
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)


        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Assuming LpLoss is a loss function defined elsewhere
        myloss = LpLoss(size_average=False)

        # Set the reporting interval (e.g., every 10 epochs)
        REPORT_INTERVAL = 10

        # Wrap the range with tqdm to get the progress bar object
        pbar = tqdm(range(epochs), desc="FNO Training")

        for epoch in pbar:
            self.model.train() # Set the model to training mode

            # Training passes
            train_loss = 0
            for x, y in train_loader:
                # Data is already on device from DataLoader
                # x, y = x.to(self.device), y.to(self.device) # This line is redundant

                optimizer.zero_grad()
                out = self.model(x)

                # Adjust view based on the output shape of your FNO1d and y
                loss = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
                loss.backward() # use the l2 relative loss

                optimizer.step()
                train_loss += loss.item()
            # Calculate average training loss per sample (since LpLoss size_average=False sums)
            train_loss /= len(train_loader.dataset)
            scheduler.step()

            # Training/Testing eval
            self.model.eval() # Set the model to evaluation mode
            with torch.no_grad():
                # Pass the full tensors for evaluation to _callback
                # X_train and X_test are already on self.device
                y_pred_train = self.model(X_train)
                y_pred_test = self.model(X_test)

                # Call the callback function to calculate and store metrics
                # Assuming y_pred_train and y_pred_test have shape [batch_size, spatial_dim, output_dim]
                # and you're taking the first output_dim (index 0)
                self._callback(y_pred_train[:,:,0], y_train, y_pred_test[:,:,0], y_test, epoch, save_loc, save_name)

                # --- Update tqdm postfix periodically ---
                # Access the last calculated NRMSE values from the stored lists
                current_train_nrmse = self.train_nrmse[-1] if self.train_nrmse else float('nan')
                current_test_nrmse = self.test_nrmse[-1] if self.test_nrmse else float('nan')


                if epoch % REPORT_INTERVAL == 0 or epoch == epochs - 1:
                     pbar.set_postfix({
                        'Train NRMSE': f'{current_train_nrmse:.8f}',
                        'Test NRMSE': f'{current_test_nrmse:.8f}',
                        'Best Test NRMSE': f'{self.best_nrmse.item():.8f}' if self.best_nrmse is not None else 'N/A' # Display best NRMSE
                    })

            # The original printing logic is commented out, rely on tqdm postfix

        # Print final best metrics after the loop finishes
        print(f"\nTraining finished.")
        print(f"Best Test NRMSE: {self.best_nrmse.item():.8f} at epoch {self.best_epoch}" if self.best_nrmse is not None else "Best Test NRMSE: N/A")


    def _callback(self, y_pred_train, y_train, y_pred_test, y_test, epoch, save_loc, save_name):
        # Ensure tensors are on the same device for calculations if they weren't already
        # This might be redundant if all inputs to _callback are guaranteed to be on self.device
        # y_pred_train = y_pred_train.to(self.device)
        # y_train = y_train.to(self.device)
        # y_pred_test = y_pred_test.to(self.device)
        # y_test = y_test.to(self.device)

        # NRMSE
        # Added small epsilon to avoid division by zero
        epsilon = 1e-8
        tr_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_train - y_train))) / torch.sqrt(torch.square(y_train).mean() + epsilon)
        te_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_test - y_test))) / torch.sqrt(torch.square(y_test).mean() + epsilon)

        # NMSE
        # Added small epsilon to avoid division by zero
        tr_nmse = torch.mean(torch.square((y_pred_train - y_train)/ (y_train + epsilon)))
        te_nmse = torch.mean(torch.square((y_pred_test - y_test) / (y_test + epsilon)))


        # Saving metrics (store item() to avoid keeping tensors in lists)
        self.train_nrmse.append(tr_nrmse.item())
        self.test_nrmse.append(te_nrmse.item())
        self.train_nmse.append(tr_nmse.item())
        self.test_nmse.append(te_nmse.item())

        # Model Saving based on Test NRMSE
        if self.best_nrmse is None or self.best_nrmse > te_nrmse:
            self.best_nrmse = te_nrmse # Store the tensor here for comparison
            self.best_epoch = epoch
            # Save the best test prediction (clone and detach)
            self.y_pred_test_best = y_pred_test.clone().detach()
            torch.save(self.model.state_dict(), save_loc + "fno_{}.pt".format(save_name))

        # The _callback function doesn't need to return anything based on its usage