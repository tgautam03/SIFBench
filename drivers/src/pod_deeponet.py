import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming Adam is imported from torch.optim
from torch.optim import Adam
import itertools # This is used in the train method, make sure it's imported
from tqdm import tqdm # Make sure tqdm is imported

# Provided Net class
class Net(nn.Module):
    def __init__(self, layers, act=nn.Tanh()):
        super(Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight) # Initialize weights
        
    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x) # Last layer has no activation
        return x

# Provided LpLoss class (assuming it's the same as before and correctly defined)
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


# Modified POD_DeepONet class inheriting from nn.Module and adding tqdm postfix updates
class POD_DeepONet(torch.nn.Module): # <--- Added inheritance from torch.nn.Module
    def __init__(self, y_train, branch_layers, K, device, testing=False, branch_net_loc=None):
        super(POD_DeepONet, self).__init__() # <--- Added call to the parent class constructor
        self.device = device

        # Branch Net (assuming Net is an nn.Module)
        self.branch_net = Net(branch_layers + [K], act=nn.Tanh()).to(self.device)
        if testing:
            if device == "cpu":
                # Added weights_only=False for compatibility
                self.branch_net.load_state_dict(torch.load(branch_net_loc, map_location=torch.device('cpu'), weights_only=False))
            else:
                # Added weights_only=False
                self.branch_net.load_state_dict(torch.load(branch_net_loc, weights_only=False))

        # Trunk Net (fixed basis from SVD of training data)
        # Ensure y_train is on the correct device for SVD
        y_train_device = y_train.to(self.device)
        U, S, B = torch.linalg.svd(y_train_device.reshape([len(y_train_device),-1]), full_matrices=False)
        self.B = B[0:K] #K x D - keep this on the device
        del y_train_device # Free up memory

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
    def forward(self, branch_in):
        # Ensure branch_in is on the correct device
        branch_pred = self.branch_net(branch_in.to(self.device)) # N x K
        trunk_pred = self.B # K x D^2 (already on device)
        y = torch.matmul(branch_pred, trunk_pred) # N x D^2
        return y

    # Renamed the train method to avoid clashing with nn.Module.train (optional but good practice)
    def train_model(self, branch_in_train, y_train, branch_in_test, y_test, lr, epochs, save_loc, save_name):
        # Move data to device once before the training loop
        branch_in_train = branch_in_train.to(self.device)
        y_train = y_train.to(self.device)
        branch_in_test = branch_in_test.to(self.device)
        y_test = y_test.to(self.device)

        params = list(itertools.chain(self.branch_net.parameters()))
        minimizer = Adam(params, lr=lr)
        
        # Set the reporting interval (e.g., every 100 epochs)
        REPORT_INTERVAL = 100

        # Wrap the range with tqdm to get the progress bar object
        pbar = tqdm(range(epochs), desc="POD-DeepONet Training")

        for epoch in pbar:
            # Set the model to training mode (using the inherited nn.Module.train method)
            self.train() # This calls nn.Module.train()
            # Or explicitly: super().train()

            minimizer.zero_grad()

            # Training pass
            # forward method handles moving branch_in_train to device
            y_pred_train = self.forward(branch_in_train)
            # Ensure shapes match for loss calculation
            loss = (y_pred_train.reshape(y_train.shape) - y_train).square().mean()

            # Training/Testing eval
            # Set the model to evaluation mode (using the inherited nn.Module.eval method)
            self.eval() # This calls nn.Module.eval()
            # Or explicitly: super().eval()

            with torch.no_grad():
                # Test preds
                 # forward method handles moving branch_in_test to device
                y_pred_test = self.forward(branch_in_test)
                # Pass tensors which are already on self.device to _callback
                self._callback(y_pred_train, y_train, y_pred_test, y_test, epoch, save_loc, save_name)

                # --- Update tqdm postfix periodically ---
                # Access the last calculated metrics from the stored lists
                current_train_nrmse = self.train_nrmse[-1] if self.train_nrmse else float('nan')
                current_test_nrmse = self.test_nrmse[-1] if self.test_nrmse else float('nan')
                current_train_nmse = self.train_nmse[-1] if self.train_nmse else float('nan')
                current_test_nmse = self.test_nmse[-1] if self.test_nmse else float('nan')
                best_test_nrmse_val = self.best_nrmse.item() if self.best_nrmse is not None else float('nan')


                if epoch % REPORT_INTERVAL == 0 or epoch == epochs - 1:
                     pbar.set_postfix({
                        'Tr NRMSE': f'{current_train_nrmse:.8f}',
                        'Te NRMSE': f'{current_test_nrmse:.8f}',
                        'Tr NMSE': f'{current_train_nmse:.8f}',
                        'Te NMSE': f'{current_test_nmse:.8f}',
                        'Best Te NRMSE': f'{best_test_nrmse_val:.8f}' if not torch.isnan(torch.tensor(best_test_nrmse_val)) else 'N/A' # Display best NRMSE
                    })


            # Training update (outside the no_grad block)
            loss.backward()
            minimizer.step()

        # Print final best metrics after the loop finishes
        print(f"\nTraining finished.")
        print(f"Best Test NRMSE: {self.best_nrmse.item():.8f} at epoch {self.best_epoch}" if self.best_nrmse is not None else "Best Test NRMSE: N/A")


    def _callback(self, y_pred_train, y_train, y_pred_test, y_test, epoch, save_loc, save_name):
        # Ensure tensors are on the same device for calculations if they weren't already
        # (They should be if the train_model method moves them before calling _callback)
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
            torch.save(self.branch_net.state_dict(), save_loc + "pod_branch_{}.pt".format(save_name))

        # The _callback function doesn't need to return anything based on its usage