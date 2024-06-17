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
    'test': ('../data/pre_processed/cnn_data/X_test.csv', '../data/pre_processed/cnn_data/y_test.csv')
}


def load_and_prepare_data(x_path: str, y_path: str) -> tuple:
    """
    Load and prepare data from CSV files.

    :param x_path: Path to the features CSV file.
    :param y_path: Path to the target CSV file.
    :return: Tuple of tensors (X, y).
    """
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)
    return X_tensor, y_tensor


# Load the data
X_train_tensor, y_train_tensor = load_and_prepare_data(*file_paths['train'])
X_test_tensor, y_test_tensor = load_and_prepare_data(*file_paths['test'])

# Check data balance
print("Train target distribution:\n", y_train_tensor.cpu().numpy().flatten())
print("Test target distribution:\n", y_test_tensor.cpu().numpy().flatten())


class DataSet(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Initialize the dataset.

        :param X: Feature tensor.
        :param y: Target tensor.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        :return: Length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item from the dataset.

        :param idx: Index of the item.
        :return: Tuple of feature and target tensors.
        """
        return self.X[idx], self.y[idx]


class CNN(nn.Module):
    def __init__(self):
        """
        Initialize the CNN model.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=2, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(40, 1)  # Output layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))  # Ensure non-negative outputs with ReLU
        return x


class ModelTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int,
                 device: torch.device):
        """
        Initialize the model trainer.

        :param model: The model to train.
        :param criterion: The loss function.
        :param optimizer: The optimizer.
        :param num_epochs: Number of epochs to train.
        :param device: Device to train on (CPU or GPU).
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path_loss: str, save_path_performance: str,
              save_path_model: str):
        """
        Train the model.

        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param save_path_loss: Path to save the loss plot.
        :param save_path_performance: Path to save the performance metrics.
        :param save_path_model: Path to save the trained model.
        """
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

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if save_path_loss:
            self.plot_loss(self.train_losses, self.val_losses, save_path_loss)

        if save_path_model:
            self.save_model(save_path_model)

    def evaluate_model(self, data_loader: DataLoader) -> tuple:
        """
        Evaluate the model.

        :param data_loader: DataLoader for the evaluation data.
        :return: Tuple of loss, outputs, and targets.
        """
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

    def plot_loss(self, train_losses: list, val_losses: list, save_path: str = None):
        """
        Plot the training and validation loss.

        :param train_losses: List of training losses.
        :param val_losses: List of validation losses.
        :param save_path: Path to save the plot.
        """
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

    def save_model(self, save_path: str):
        """
        Save the trained model.

        :param save_path: Path to save the model.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        existing_files = os.listdir(save_path)
        model_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith('model')]
        new_model_number = max(model_numbers) + 1 if model_numbers else 1
        file_name = f'model_cnn_{new_model_number}.pth'
        file_path = os.path.join(save_path, file_name)
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')


def plot_predictions(targets: np.ndarray, outputs: np.ndarray):
    """
    Plot the targets and outputs.

    :param targets: Actual target values.
    :param outputs: Predicted output values.
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


def plot_distribution(targets: np.ndarray, outputs: np.ndarray):
    """
    Plot the distribution of targets and outputs with the same bin widths.

    :param targets: Actual target values.
    :param outputs: Predicted output values.
    """
    min_val = min(np.min(targets), np.min(outputs))
    max_val = max(np.max(targets), np.max(outputs))

    # Calculate the number of bins based on the range of values and desired bin width
    binwidth = (max_val - min_val) / 50  # Adjust the bin width as needed
    bins = np.arange(min_val, max_val + binwidth, binwidth)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(targets, bins=bins, color='blue', alpha=0.5, label='Targets')
    ax[1].hist(outputs, bins=bins, color='red', alpha=0.5, label='Outputs')
    ax[0].set_title('Target Distribution')
    ax[1].set_title('Output Distribution')
    ax[0].set_xlim(min_val, max_val)
    ax[1].set_xlim(min_val, max_val)
    ax[0].legend()
    ax[1].legend()
    plt.show()


def cross_validate_model(model_class: type, X_tensor: torch.Tensor, y_tensor: torch.Tensor, n_splits: int = 5,
                         epochs: int = 25, lr: float = 0.0001, **kwargs):
    """
    Perform cross-validation on the model.

    :param model_class: The model class to instantiate.
    :param X_tensor: Feature tensor.
    :param y_tensor: Target tensor.
    :param n_splits: Number of folds for cross-validation.
    :param epochs: Number of epochs for training.
    :param lr: Learning rate for the optimizer.
    :param kwargs: Additional arguments for ModelTrainer.
    """
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
        trainer.train(train_loader, val_loader, kwargs.get('save_path_loss'), kwargs.get('save_path_performance'),
                      kwargs.get('save_path_model'))

        val_loss, _, _ = trainer.evaluate_model(val_loader)
        mse_scores.append(val_loss)

    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f'Cross-Validated MSE: {avg_mse:.4f} ± {std_mse:.4f}')


def rescale(scaler_path: str, test_outputs: np.ndarray, test_targets: np.ndarray) -> tuple:
    """
    Rescale the test outputs and targets using a saved scaler object.

    :param scaler_path: Path to the saved scaler object.
    :param test_outputs: Predicted output values.
    :param test_targets: Actual target values.
    :return: Tuple of rescaled outputs and targets.
    """
    scaler = joblib.load(scaler_path)
    test_outputs = scaler.inverse_transform(test_outputs)
    test_targets = scaler.inverse_transform(test_targets)
    return test_outputs, test_targets


def main():
    """
    Main function to run the training, evaluation, and plotting.
    """
    # Prepare datasets and dataloaders
    test_dataset = DataSet(X_test_tensor, y_test_tensor)
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

    cross_validate_model(CNN, X_train_tensor, y_train_tensor, epochs=7, n_splits=5, save_path_model=save_path_model,
                         save_path_loss=save_path_loss, save_path_performance=save_path_performance)

    # Evaluate on the test set
    test_loss, test_outputs, test_targets = trainer.evaluate_model(test_loader)

    # Rescale the outputs and targets
    test_outputs, test_targets = rescale('../data/scaler/energy_scaler.pkl', test_outputs, test_targets)

    plot_predictions(test_targets, test_outputs)
    plot_distribution(test_targets, test_outputs)

    print(f'Test Loss: {test_loss:.4f}')
    print('Test Targets:', test_targets)
    print('Test Outputs:', test_outputs)


if __name__ == '__main__':
    main()
