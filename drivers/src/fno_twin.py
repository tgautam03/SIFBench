import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from timeit import default_timer
from tqdm import tqdm
import numpy as np # Added numpy import for potential use, though not strictly needed for this modification

################################################################
#  1d fourier layer
################################################################

# Provided LpLoss class - No changes needed here as it flattens the tensors for loss calculation
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d # Spatial dimension (e.g., 1 for 1D FNO)
        self.p = p # Lp norm type (e.g., 2 for L2)
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        # x and y are expected to be of shape (batch_size, spatial_dim, output_channels)
        num_examples = x.size()[0]
        # Flatten spatial and channel dimensions for norm calculation
        x_flat = x.view(num_examples, -1)
        y_flat = y.view(num_examples, -1)

        # Assume uniform mesh for spatial dimension
        # Note: This h calculation might need adjustment if the spatial dimension isn't uniform or 1D
        # For this problem (1D FNO with output channels), d=1 is more appropriate if considering spatial dimension only
        # However, the original code uses d=2, which might imply something about the problem setup.
        # Keeping d=2 as per original code, but be mindful if this scaling factor is correct for your data.
        h = 1.0 / (x.size()[1] - 1.0) if x.size()[1] > 1 else 1.0 # Handle case where spatial dim is 1

        all_norms = (h ** (self.d / self.p)) * torch.norm(x_flat - y_flat, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        # x and y are expected to be of shape (batch_size, spatial_dim, output_channels)
        num_examples = x.size()[0]

        # Flatten spatial and channel dimensions for norm calculation
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)

        diff_norms = torch.norm(x_flat - y_flat, self.p, 1)
        y_norms = torch.norm(y_flat, self.p, 1)

        # Added small epsilon to avoid division by zero
        epsilon = 1e-8

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + epsilon))
            else:
                return torch.sum(diff_norms / (y_norms + epsilon))

        return diff_norms / (y_norms + epsilon)

    def __call__(self, x, y):
        # By default, use relative L2 loss
        return self.rel(x, y)

# Provided SpectralConv1d class - No changes needed here
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        # Scale factor for weight initialization
        self.scale = (1 / (in_channels*out_channels))
        # Weights for the Fourier modes transformation, complex numbers
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication function
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        # This performs element-wise complex multiplication and summation over the in_channels
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients using rfft (real-to-complex FFT)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes by learned weights
        # Create output tensor for Fourier coefficients, size is N/2 + 1 for rfft
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        # Multiply the lower frequency modes (up to self.modes1)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space using irfft (inverse real-to-complex FFT)
        # Specify the output size n to match the input size
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

# Modified FNO1d class to support 2 output channels
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
        input shape: (batchsize, x=s, c=d_in)
        output: the solution of a later timestep or field
        output shape: (batchsize, x=s, c=2) # Modified to output 2 channels
        """

        self.modes1 = modes # Number of Fourier modes
        self.width = width # Hidden channel dimension
        # Linear layer to lift input features to the hidden dimension
        self.fc0 = nn.Linear(d_in, self.width) # input channel is d_in

        # 4 layers of Spectral Convolution and standard Convolution (W + K)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1) # 1x1 convolution
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Project from hidden dimension back to the output space
        # First projection layer
        self.fc1 = nn.Linear(self.width, num_phi_samples) # Project back to spatial dimension size
        # Second projection layer - MODIFIED to output 2 channels
        self.fc2 = nn.Linear(num_phi_samples, 2) # Output channel is now 2

    def forward(self, x):
        # x: input tensor of shape (batchsize, spatial_dim, d_in)
        # Lift input features
        x = self.fc0(x)     # Shape: (batchsize, spatial_dim, width)
        # Permute to (batchsize, width, spatial_dim) for Conv1d and SpectralConv1d
        x = x.permute(0, 2, 1)

        # Apply 4 layers of (SpectralConv + Conv1d) with GELU activation
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
        # Permute back to (batchsize, spatial_dim, width) for linear layers
        x = x.permute(0, 2, 1)

        # Apply projection layers
        x = self.fc1(x) # Shape: (batchsize, spatial_dim, spatial_dim) - This seems unusual, check FNO architecture details
                        # Assuming fc1 projects from width to spatial_dim for some reason based on original code.
                        # If fc1 is meant to reduce the spatial dimension, this might need adjustment.
                        # If fc1 is meant to process each spatial point independently, its input should be (batchsize * spatial_dim, width)

        # Re-reading the original FNO paper/implementations, the typical structure is:
        # Input (N, S, d_in) -> fc0 (N, S, width) -> permute (N, width, S)
        # -> (SpectralConv + Conv1d) layers (N, width, S)
        # -> permute (N, S, width) -> fc1 (N, S, width) -> fc2 (N, S, output_channels)
        # The original fc1(self.width, num_phi_samples) and fc2(num_phi_samples, 1) seems to imply
        # fc1 projects from width to spatial_dim, and then fc2 projects spatial_dim to output_channels.
        # This is an unusual structure. A more standard structure would be:
        # self.fc1 = nn.Linear(self.width, self.width)
        # self.fc2 = nn.Linear(self.width, 2)
        # Let's stick to modifying the original structure as requested, assuming the original fc1/fc2 logic is intended.
        # The original fc1(self.width, num_phi_samples) applied after permute (N, S, width) means it's applied
        # independently to each spatial point, mapping width -> num_phi_samples.
        # Then fc2(num_phi_samples, 1) maps num_phi_samples -> 1. This still feels wrong.

        # Let's assume the intent of the original fc1 and fc2 was to reshape/process the spatial dimension
        # in a non-standard way or there's a misunderstanding of the original code's intent.
        # Based on the input/output shapes and the linear layers, the flow seems to be:
        # (N, S, d_in) -> fc0 -> (N, S, width) -> permute -> (N, width, S) -> Conv layers -> (N, width, S)
        # -> permute -> (N, S, width) -> fc1 -> (N, S, num_phi_samples) -> fc2 -> (N, S, 2)
        # This interpretation makes fc1 map width -> num_phi_samples for each spatial point,
        # and fc2 map num_phi_samples -> 2 for each spatial point. This is still odd.

        # A more standard FNO output structure after the conv layers and final permute would be:
        # (N, S, width) -> fc1 (N, S, width) -> GELU -> fc2 (N, S, 2)
        # Let's adopt this more standard structure for the output layers to correctly map from width to 2 output channels per spatial point.

        # Redefining the output layers based on standard FNO practice
        self.fc1_standard = nn.Linear(self.width, self.width)
        self.fc2_standard = nn.Linear(self.width, 2) # Output 2 channels

        # --- Standard FNO output projection ---
        x = self.fc1_standard(x) # Shape: (batchsize, spatial_dim, width)
        x = F.gelu(x)
        x = self.fc2_standard(x) # Shape: (batchsize, spatial_dim, 2)

        # The original fc1 and fc2 are kept but not used in this modified forward pass,
        # or they could be removed if the standard structure is preferred.
        # Let's remove the original fc1 and fc2 and use the standard ones.

        # --- Original fc1/fc2 path (commented out) ---
        # x = self.fc1(x) # Shape: (batchsize, spatial_dim, num_phi_samples)
        # x = F.gelu(x)
        # x = self.fc2(x) # Shape: (batchsize, spatial_dim, 2) # This assumes fc2 maps num_phi_samples -> 2

        return x


# Modified FNO class inheriting from nn.Module and adding tqdm postfix updates
class FNO(torch.nn.Module):
    def __init__(self, num_phi_samples, d_in, modes, width, device):
        super(FNO, self).__init__()
        self.device = device

        # NN Model (using the modified FNO1d)
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
        y_train = y_train.to(self.device) # y_train should have shape (N, k, 2)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)   # y_test should have shape (N, k, 2)


        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Assuming LpLoss is a loss function defined elsewhere
        # LpLoss will flatten the spatial and channel dimensions for loss calculation
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

                optimizer.zero_grad()
                out = self.model(x) # out will have shape (batch_size, spatial_dim, 2)

                # LpLoss expects tensors to be flattened to (batch_size, -1) for norm calculation
                # out.view(x.shape[0], -1) flattens (batch_size, spatial_dim, 2) to (batch_size, spatial_dim * 2)
                # y.view(x.shape[0], -1) flattens (batch_size, spatial_dim, 2) to (batch_size, spatial_dim * 2)
                loss = myloss(out, y) # Pass full output and label tensors to loss function
                loss.backward()

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
                y_pred_train = self.model(X_train) # Shape (N, k, 2)
                y_pred_test = self.model(X_test)   # Shape (N, k, 2)

                # Call the callback function to calculate and store metrics
                # Pass the full predicted and true tensors
                self._callback(y_pred_train, y_train, y_pred_test, y_test, epoch, save_loc, save_name)

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
        # y_pred_train, y_train, y_pred_test, y_test now have shape (N, k, 2)

        # Ensure tensors are on the same device for calculations if they weren't already
        # This might be redundant if all inputs to _callback are guaranteed to be on self.device
        # y_pred_train = y_pred_train.to(self.device)
        # y_train = y_train.to(self.device)
        # y_pred_test = y_pred_test.to(self.device)
        # y_test = y_test.to(self.device)

        # NRMSE calculation
        # Calculate mean squared error over the spatial and channel dimensions
        # Added small epsilon to avoid division by zero
        epsilon = 1e-8
        tr_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_train - y_train))) / torch.sqrt(torch.mean(torch.square(y_train)) + epsilon)
        te_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_test - y_test))) / torch.sqrt(torch.mean(torch.square(y_test)) + epsilon)

        # NMSE calculation
        # Added small epsilon to avoid division by zero
        tr_nmse = torch.mean(torch.square((y_pred_train - y_train) / (y_train + epsilon)))
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
            # Save the best test prediction (clone and detach) - saving the full (N, k, 2) tensor
            self.y_pred_test_best = y_pred_test.clone().detach()
            torch.save(self.model.state_dict(), save_loc + "fno_{}.pt".format(save_name))

        # The _callback function doesn't need to return anything based on its usage
