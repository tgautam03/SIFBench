import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import itertools

from tqdm import tqdm

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
    
class DeepONet(torch.nn.Module): # <--- Added inheritance from torch.nn.Module
    def __init__(self, branch_layers, trunk_layers, K, device, testing=False, branch_net_loc=None, trunk_net_loc=None):
        super(DeepONet, self).__init__() # <--- Added call to the parent class constructor
        self.device = device

        # Branch Net
        # Assuming Net is also an nn.Module
        self.branch_net = Net(branch_layers + [K], act=nn.Tanh()).to(self.device)
        if testing:
            if device == "cpu":
                # Added weights_only=False for compatibility
                self.branch_net.load_state_dict(torch.load(branch_net_loc, map_location=torch.device('cpu'), weights_only=False))
            else:
                # Added weights_only=False
                self.branch_net.load_state_dict(torch.load(branch_net_loc, weights_only=False))

        # Trunk Net
        # Assuming Net is also an nn.Module
        self.trunk_net = Net(trunk_layers + [K], act=nn.Tanh()).to(self.device)
        if testing:
            if device == "cpu":
                 # Added weights_only=False
                self.trunk_net.load_state_dict(torch.load(trunk_net_loc, map_location=torch.device('cpu'), weights_only=False))
            else:
                 # Added weights_only=False
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

    # The forward method is a standard part of nn.Module
    def forward(self, branch_in, trunk_in):
        branch_pred = self.branch_net(branch_in) # N x K
        trunk_pred = self.trunk_net(trunk_in) # D^2 x K
        # Assuming branch_pred shape N x K and trunk_pred shape D^2 x K
        # and you want to compute N x D^2 output
        # The original matmul with transpose seems correct for this
        y = torch.matmul(branch_pred, trunk_pred.transpose(0, 1))
        return y


    def train_model(self, branch_in_train, trunk_in_train, y_train, branch_in_test, trunk_in_test, y_test, lr, epochs, save_loc, save_name):
        # Note: This method name clashes with the nn.Module.train() method.
        # While it works after inheriting from nn.Module (self.train() inside
        # this method will call the nn.Module one), it can be confusing.
        # Consider renaming this method, e.g., `train_model`.

        params = list(itertools.chain(self.branch_net.parameters(), self.trunk_net.parameters()))
        minimizer = torch.optim.Adam(params, lr=lr)
        epoch_tolerance = 0

        # Set the reporting interval (e.g., every 100 epochs)
        REPORT_INTERVAL = 100

        # Wrap the range with tqdm to get the progress bar object
        pbar = tqdm(range(epochs), desc="DeepONet Training")

        for epoch in pbar:
            super().train() # <--- Correctly call the parent class's train() method
            # Or simply: self.train() # This will now call nn.Module.train() after inheritance

            minimizer.zero_grad()

            # Training pass
            y_pred_train = self.forward(branch_in_train, trunk_in_train)
            # Ensure shapes match for loss calculation if y_train is not already reshaped
            loss = (y_pred_train.reshape(y_train.shape) - y_train).square().mean()

            # Training/Testing eval
            super().eval() # <--- Correctly call the parent class's eval() method
            # Or simply: self.eval() # This will now call nn.Module.eval() after inheritance

            with torch.no_grad():
                # Test preds
                y_pred_test = self.forward(branch_in_test, trunk_in_test)

                # Ensure shapes match for metric calculations
                y_pred_train_reshaped = y_pred_train.reshape(y_train.shape)
                y_pred_test_reshaped = y_pred_test.reshape(y_test.shape)


                # NRMSE
                tr_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_train_reshaped - y_train))) / torch.sqrt(torch.square(y_train).mean())
                te_nrmse = torch.sqrt(torch.mean(torch.square(y_pred_test_reshaped - y_test))) / torch.sqrt(torch.square(y_test).mean())

                # NMSE
                # Added small epsilon to avoid division by zero if y_train or y_test contains zeros
                epsilon = 1e-8
                tr_nmse = torch.mean(torch.square((y_pred_train_reshaped - y_train) / (y_train + epsilon)))
                te_nmse = torch.mean(torch.square((y_pred_test_reshaped - y_test) / (y_test + epsilon)))


                # Saving metrics for plotting later
                self.train_nrmse.append(tr_nrmse.item())
                self.test_nrmse.append(te_nrmse.item())
                self.train_nmse.append(tr_nmse.item())
                self.test_nmse.append(te_nmse.item())

                # --- Update tqdm postfix periodically ---
                if epoch % REPORT_INTERVAL == 0 or epoch == epochs - 1:
                     pbar.set_postfix({
                        'Train NRMSE': f'{tr_nrmse.item():.4f}',
                        'Test NRMSE': f'{te_nrmse.item():.4f}',
                        'Train NMSE': f'{tr_nmse.item():.4f}',
                        'Test NMSE': f'{te_nmse.item():.4f}',
                        'Tolerance': epoch_tolerance
                    })


                # Model Saving based on Test NRMSE
                if self.best_nrmse is None or self.best_nrmse > te_nrmse:
                    self.best_nrmse = te_nrmse
                    self.best_epoch = epoch
                    self.y_pred_test_best = y_pred_test.clone().detach() # Save the best test prediction
                    torch.save(self.branch_net.state_dict(), save_loc + "branch_{}.pt".format(save_name))
                    torch.save(self.trunk_net.state_dict(), save_loc + "trunk_{}.pt".format(save_name))
                    epoch_tolerance = 0 # Reset tolerance
                elif self.best_nrmse <= te_nrmse: # Tolerance increases only if performance does not improve
                    epoch_tolerance += 1

                # --- Early Stopping Criteria ---
                # Note: Your original early stopping tolerance (50000) and
                # minimum epochs (100000) are very high.
                # Consider if these are the intended values.
                # Also, the condition for breaking should probably be
                # based on `epoch_tolerance >= threshold` and `epoch >= min_epochs_before_stopping`.
                EARLY_STOP_TOLERANCE = 50000 # Define your tolerance threshold
                MIN_EPOCHS_BEFORE_STOPPING = 100000 # Define minimum epochs before considering stopping

                if epoch_tolerance >= EARLY_STOP_TOLERANCE and epoch >= MIN_EPOCHS_BEFORE_STOPPING:
                     print("\nExiting loop due to early stopping...")
                     print(f"Best Test NRMSE: {self.best_nrmse:.8f} at epoch {self.best_epoch}")
                     break


            # Training update (outside the no_grad block)
            loss.backward()
            minimizer.step()

        # Print final best metrics after the loop finishes
        print(f"\nTraining finished.")
        print(f"Best Test NRMSE: {self.best_nrmse:.8f} at epoch {self.best_epoch}")

        return