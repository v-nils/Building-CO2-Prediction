import joblib
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

if cuda_available:
    print("CUDA Available. Using", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

# File paths
file_paths = {
    'train': ('../data/pre_processed/cnn_data/X_train.csv', '../data/pre_processed/cnn_data/y_train.csv'),
    'val': ('../data/pre_processed/cnn_data/X_val.csv', '../data/pre_processed/cnn_data/y_val.csv'),
    'test': ('../data/pre_processed/cnn_data/X_test.csv', '../data/pre_processed/cnn_data/y_test.csv')
}

# Load the data
def load_and_prepare_data(x_path, y_path):
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)
    return X_tensor, y_tensor

# Load the data
X_train_tensor, y_train_tensor = load_and_prepare_data(*file_paths['train'])
X_val_tensor, y_val_tensor = load_and_prepare_data(*file_paths['val'])
X_test_tensor, y_test_tensor = load_and_prepare_data(*file_paths['test'])

# Check data balance
print("Train target distribution:\n", y_train_tensor.cpu().numpy().flatten())
print("Validation target distribution:\n", y_val_tensor.cpu().numpy().flatten())
print("Test target distribution:\n", y_test_tensor.cpu().numpy().flatten())

# Dataset class
class DataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7, 1)  # Output layer without activation function

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)  # Direct output without activation function
        return x

# Model Trainer
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, num_epochs, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device

    def train(self, train_loader, val_loader, save_path_loss, save_path_performance, save_path_model):
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_train_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            train_loss = running_train_loss / len(train_loader.dataset)
            val_loss, val_outputs, val_targets = self.evaluate_model(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if train_losses is not None and val_losses is not None and save_path_loss is not None:
            self.plot_loss(train_losses, val_losses, save_path_loss)

        if save_path_model is not None:
            self.save_model(save_path_model)

    def evaluate_model(self, data_loader):
        self.model.eval()
        loss = 0.0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss += self.criterion(outputs, targets).item() * inputs.size(0)
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        loss /= len(data_loader.dataset)
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        return loss, all_outputs, all_targets

    def plot_loss(self, train_losses, val_losses, save_path=None):
        file_name = os.path.join(save_path, 'loss_plot.png')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label='Training Loss', marker='^', linestyle='--', linewidth=0.75, markersize=1)
        ax.plot(val_losses, label='Validation Loss', marker='x', linestyle='-.', linewidth=0.75, markersize=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training and Validation Loss')

        if save_path:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        existing_files = os.listdir(save_path)
        model_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith('model')]
        new_model_number = max(model_numbers) + 1 if model_numbers else 1
        file_name = f'model_cnn_{new_model_number}.pth'
        file_path = os.path.join(save_path, file_name)
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')

def plot_predictions(targets, outputs):
    """
    Plot the targets and outputs.
    :param targets:
    :param outputs:
    :return:
    """

    min_val = min(np.min(targets), np.min(outputs))
    max_val = max(np.max(targets), np.max(outputs))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(targets, outputs, label='Targets vs Outputs', color='blue', marker='^')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title('Targets vs Outputs')
    plt.show()


def plot_distribution(targets, outputs):
    """
    Plot the distribution of targets and outputs.
    :param targets:
    :param outputs:
    :return:
    """

    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    ax[0].hist(targets, bins=50, color='blue', alpha=0.5, label='Targets')
    ax[1].hist(outputs, bins=50, color='red', alpha=0.5, label='Outputs')
    ax[0].set_title('Target Distribution')
    ax[1].set_title('Output Distribution')
    plt.show()


def cross_validate_model(model_class, X_tensor, y_tensor, n_splits=5, epochs=25, lr=0.0001, **kwargs):
    kf = KFold(n_splits=n_splits, shuffle=True)
    mse_scores = []

    for train_index, val_index in kf.split(X_tensor):
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]

        train_dataset = DataSet(X_train, y_train)
        val_dataset = DataSet(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = model_class().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = ModelTrainer(model, criterion, optimizer, epochs, device)
        trainer.train(train_loader, val_loader,
                      kwargs.get('save_path_loss'), kwargs.get('save_path_performance'), kwargs.get('save_path_model'))

        val_loss, _, _ = trainer.evaluate_model(val_loader)
        mse_scores.append(val_loss)

    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f'Cross-Validated MSE: {avg_mse:.4f} ± {std_mse:.4f}')

def main():
    # Prepare datasets and dataloaders
    train_dataset = DataSet(X_train_tensor, y_train_tensor)
    val_dataset = DataSet(X_val_tensor, y_val_tensor)
    test_dataset = DataSet(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = CNN().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    save_path_loss = '../data/results/cnn/loss_fun'
    save_path_performance = '../data/results/cnn/performance'
    save_path_model = '../data/cnn_models'

    trainer = ModelTrainer(model, criterion, optimizer, num_epochs=25, device=device)
    #trainer.train(train_loader, val_loader, save_path_loss, save_path_performance, save_path_model)

    cross_validate_model(CNN, X_train_tensor, y_train_tensor, n_splits=5, save_path_model=save_path_model, save_path_loss=save_path_loss, save_path_performance=save_path_performance)

    # Evaluate on the test set
    test_loss, test_outputs, test_targets = trainer.evaluate_model(test_loader)

    test_outputs, test_targets = rescale('../data/scaler/energy_scaler.pkl', test_outputs, test_targets)

    plot_predictions(test_targets, test_outputs)

    print(f'Test Loss: {test_loss:.4f}')
    print('Test Targets:', test_targets)
    print('Test Outputs:', test_outputs)

def rescale(scaler_path: str, test_outputs, test_targets) -> tuple:
    """
    Rescale the test outputs and targets using a saved scaler object.
    :param scaler_path: (str) Path to the saved scaler object
    :return: None
    """
    scaler = joblib.load(scaler_path)
    test_outputs = scaler.inverse_transform(test_outputs)
    test_targets = scaler.inverse_transform(test_targets)
    return test_outputs, test_targets



if __name__ == '__main__':
    main()
