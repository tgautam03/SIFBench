import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class Net10(nn.Module):
    def __init__(self, inp_feat, num_neurons):
        super(Net10, self).__init__()
        self.fc0 = nn.Linear(inp_feat, num_neurons)
        self.fc1 = nn.Linear(num_neurons, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, num_neurons)
        self.fc4 = nn.Linear(num_neurons, num_neurons)
        self.fc5 = nn.Linear(num_neurons, num_neurons)
        self.fc6 = nn.Linear(num_neurons, num_neurons)
        self.fc7 = nn.Linear(num_neurons, num_neurons)
        self.fc8 = nn.Linear(num_neurons, num_neurons)
        self.fc9 = nn.Linear(num_neurons, num_neurons)
        self.fc10 = nn.Linear(num_neurons, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))
        x = F.leaky_relu(self.fc7(x))
        x = F.leaky_relu(self.fc8(x))
        x = F.leaky_relu(self.fc9(x))
        x = self.fc10(x)
        return x

def train_model(net, X_train, y_train, X_val, y_val, loss_fn, optimizer, EPOCHS, FILENAME):
    """
    Trains a PyTorch model on the full dataset per iteration and
    displays train/val MSE periodically in the tqdm bar.

    Args:
        net: The PyTorch model to train.
        X_train: Training features (torch.Tensor).
        y_train: Training target (torch.Tensor).
        X_val: Validation features (torch.Tensor).
        y_val: Validation target (torch.Tensor).
        loss_fn: The loss function (e.g., torch.nn.MSELoss).
        optimizer: The optimizer.
        EPOCHS: The total number of training iterations (epochs).
        FILENAME: Base filename for saving the best model.
    """
    train_mse = []
    val_mse = []
    epoch_tolerance = 0
    best_val_loss = float('inf') # Initialize with a large value

    # Set the reporting interval (e.g., every 100 epochs)
    REPORT_INTERVAL = 100

    # Wrap the range with tqdm to get the progress bar object
    pbar = tqdm(range(EPOCHS + 1), desc="Training Progress")

    for it in pbar:
        net.train() # Set the network to training mode
        optimizer.zero_grad()

        # Forward pass on the entire training data
        y_pred = net(X_train)
        loss = loss_fn(y_pred, y_train)

        # --- Validation Phase (on entire validation set) ---
        net.eval() # Set the network to evaluation mode
        with torch.no_grad():
            y_val_pred = net(X_val)
            val_loss = loss_fn(y_val_pred, y_val)
        net.train() # Set back to training mode

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        current_train_loss = loss.item()
        current_val_loss = val_loss.item()

        # --- Update tqdm postfix periodically ---
        if it % REPORT_INTERVAL == 0 or it == EPOCHS:
            pbar.set_postfix({
                'Train MSE': f'{current_train_loss:.4f}',
                'Val MSE': f'{current_val_loss:.4f}',
                'Tolerance': epoch_tolerance
            })

        # --- Model Saving and Early Stopping ---
        if len(val_mse) == 0 or current_val_loss < best_val_loss:
             if len(val_mse) > 0: # Avoid printing on the very first iteration
                  # Optional: Print when a better model is saved
                  # print(f"\nEpoch {it}: Val loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}. Saving model.")
                  pass

             best_val_loss = current_val_loss
             torch.save(net.state_dict(), FILENAME)
             train_mse.append(current_train_loss)
             val_mse.append(current_val_loss)
             epoch_tolerance = 0
        elif current_val_loss >= best_val_loss:
             epoch_tolerance += 1
             # Only append losses if a new best model was NOT saved,
             # to keep train_mse and val_mse lists aligned with saved models.
             # If you want to record loss for every epoch, move these two lines
             # outside this if/else block, but be aware the lists
             # won't directly correspond to saved model epochs.
             train_mse.append(current_train_loss)
             val_mse.append(current_val_loss)


        # --- Early Stopping Criteria ---
        # Note: The original tolerance check seemed very high (50000 epochs).
        # Consider if this is the intended behavior or if a smaller number is appropriate.
        # Also, the condition `it >= 100000` means early stopping can only happen after 100k epochs.
        # Adjust these values as needed for your specific problem.
        if epoch_tolerance >= 50000 and it >= 100000:
            print("\nExiting loop due to early stopping...")
            # The last appended values are the ones from the epoch that triggered stopping
            print(f"Iteration: {it}; Train MSE: {train_mse[-1]:.8f}; Val MSE: {val_mse[-1]:.8f}")
            break

    # Print final losses after the loop finishes
    if len(train_mse) > 0 and len(val_mse) > 0:
        print(f"\nFinal Train MSE: {train_mse[-1]:.8f}; Final Val MSE: {val_mse[-1]:.8f}")
    else:
        print("\nTraining did not complete any epochs.")


    return train_mse, val_mse