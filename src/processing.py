import joblib
import pandas as pd
import torch
from matplotlib import gridspec
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import seaborn as sns
import os
from matplotlib.ticker import MultipleLocator
from brokenaxes import brokenaxes
from sklearn.metrics import mean_squared_error
import scienceplots

plt.style.use(['science', 'ieee'])

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

if cuda_available:
    print("CUDA Available. Using", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

# File paths
file_paths = {
    'train': '../data/pre_processed/nn_data/train.csv',
    'test': '../data/pre_processed/nn_data/test.csv',
    'all': ('../data/pre_processed/nn_data/X_all.csv', '../data/pre_processed/nn_data/y_all.csv')
}

final_results = '../data/in_out/final_results.csv'


def load_and_prepare_data(df_path) -> tuple:
    """
    Load and prepare data from CSV files.

    :param df_path: Path to the CSV file.
    :return: Tuple of tensors (X, y).
    """
    df = pd.read_csv(df_path, index_col=0)
    X_tensor = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).to(device)
    return X_tensor, y_tensor, df.index


# Load the data
X_train_tensor, y_train_tensor, train_indices = load_and_prepare_data(file_paths['train'])
X_test_tensor, y_test_tensor, test_indices = load_and_prepare_data(file_paths['test'])

# Check data balance
print("Train target distribution:\n", y_train_tensor.cpu().numpy().flatten())
print("Test target distribution:\n", y_test_tensor.cpu().numpy().flatten())


class DataSet(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, indices: pd.Index):
        """
        Initialize the dataset.

        :param X: Feature tensor.
        :param y: Target tensor.
        :param indices: Original indices of the samples.
        """
        self.X = X
        self.y = y
        self.indices = indices

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
        :return: Tuple of feature, target tensors and original index.
        """
        return self.X[idx], self.y[idx], self.indices[idx]


class NN(nn.Module):
    def __init__(self, input_shape: tuple):
        """
        Initialize the NN model.
        """
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 64, 3)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.input_shape = input_shape
        self.dropout = nn.Dropout(0.5)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)



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

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model.

        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        """

        for epoch in range(self.num_epochs):
            self.model.train()
            running_train_loss = 0.0
            for inputs, targets, _ in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            train_loss = running_train_loss / len(train_loader.dataset)
            val_loss, val_outputs, val_targets, val_indices = self.evaluate_model(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def evaluate_model(self, data_loader: DataLoader) -> tuple:
        """
        Evaluate the model.

        :param data_loader: DataLoader for the evaluation data.
        :return: Tuple of loss, outputs, targets, and original indices.
        """
        self.model.eval()
        loss = 0.0
        all_targets = []
        all_outputs = []
        all_indices = []

        with torch.no_grad():
            for inputs, targets, indices in data_loader:
                outputs = self.model(inputs)
                loss += self.criterion(outputs, targets).item() * inputs.size(0)
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
                all_indices.append(indices.cpu().numpy())

        loss /= len(data_loader.dataset)

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        all_indices = np.concatenate(all_indices)

        return loss, all_outputs, all_targets, all_indices

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


def plot_predictions(targets: np.ndarray, outputs: np.ndarray, save_path: str = None):
    """
    Plot the targets and outputs.

    :param targets: Actual target values.
    :param outputs: Predicted output values.
    :param save_path: Path to save the plot.
    """
    min_val = min(np.min(targets), np.min(outputs))
    max_val = max(np.max(targets), np.max(outputs))

    rmse = np.sqrt(mean_squared_error(targets, outputs))
    r2 = 1 - (np.sum((targets - outputs) ** 2) / np.sum((targets - np.mean(targets)) ** 2))

    fig, ax = plt.subplots()
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    ax.scatter(targets, outputs, label='Outputs', color='red', marker='x', s=5, alpha=0.6)
    ax.set_xlabel('Target', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title(r'${}^{}={}$'.format('R', 2, np.round(r2, 3)), fontsize=24)
    ax.tick_params(axis='both', colors='black', labelsize=9)
    ax.legend(fontsize=16)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_distribution(targets: np.ndarray, outputs: np.ndarray, save_path: str = None):
    """
    Plot the distribution of targets and outputs with the same bin widths and add KDE.

    :param targets: Actual target values.
    :param outputs: Predicted output values.
    :param save_path: Path to save the plot.
    """
    # Determine the bin edges for both distributions
    combined_min = min(targets.min(), outputs.min())
    combined_max = max(targets.max(), outputs.max())

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1)

    # Plot the target distribution
    sns.histplot(targets, bins=30, kde=True, ax=axes[0], color='blue', edgecolor='black')
    axes[0].set_title('Distribution of Targets')
    axes[0].set_xlim(combined_min, combined_max)

    # Plot the output distribution
    sns.histplot(outputs, bins=30, kde=True, ax=axes[1], color='green', edgecolor='black')
    axes[1].set_title('Distribution of Outputs')
    axes[1].set_xlim(combined_min, combined_max)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def cross_validate_model(model_class: type,
                         X_tensor: torch.Tensor,
                         y_tensor: torch.Tensor,
                         indices: pd.Index,
                         n_splits: int = 5,
                         epochs: int = 25,
                         lr: float = 0.0001, **kwargs):
    """
    Perform cross-validation on the model.

    :param model_class: The model class to instantiate.
    :param X_tensor: Feature tensor.
    :param y_tensor: Target tensor.
    :param indices: Original indices of the samples.
    :param n_splits: Number of folds for cross-validation.
    :param epochs: Number of epochs for training.
    :param lr: Learning rate for the optimizer.
    :param kwargs: Additional arguments for ModelTrainer.
    """

    kf = KFold(n_splits=n_splits, shuffle=True)
    mse_scores = []
    t_losses = []
    v_losses = []

    round = 0

    for train_index, val_index in kf.split(X_tensor):
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]
        train_indices, val_indices = indices[train_index], indices[val_index]

        train_dataset = DataSet(X_train, y_train, train_indices)
        val_dataset = DataSet(X_val, y_val, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = model_class(X_train.shape).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = ModelTrainer(model, criterion, optimizer, epochs, device)
        trainer.train(train_loader, val_loader)

        val_loss, _, _, _ = trainer.evaluate_model(val_loader)
        mse_scores.append(val_loss)

        t_losses.append(trainer.train_losses)
        v_losses.append(trainer.val_losses)
        round += 1

    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f'Cross-Validated MSE: {avg_mse:.4f} ± {std_mse:.4f}')
    print("Train Losses", t_losses)
    print("Val Losses", v_losses)


def rescale(scaler_path: str, test_outputs: np.ndarray, test_targets: np.ndarray) -> tuple:
    """
    Rescale the test outputs and targets using a saved scaler object.

    :param scaler_path: Path to the saved scaler object.
    :param test_outputs: Predicted output values.
    :param test_targets: Actual target values.
    :return: Tuple of rescaled outputs and targets.
    """
    scaler = joblib.load(scaler_path)
    shape = scaler.min_.shape[0] - 1

    _zero_cols = np.zeros((test_outputs.shape[0], shape))
    _test_targets = np.hstack((_zero_cols, test_targets.reshape(-1, 1)))
    _test_outputs = np.hstack((_zero_cols, test_outputs.reshape(-1, 1)))

    test_outputs = scaler.inverse_transform(_test_outputs)
    test_targets = scaler.inverse_transform(_test_targets)

    return test_outputs[:, -1], test_targets[:, -1]


def main():
    """
    Main function to run the training, evaluation, and plotting.
    """
    # Prepare datasets and dataloaders
    test_dataset = DataSet(X_test_tensor, y_test_tensor, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_dataset = DataSet(X_train_tensor, y_train_tensor, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    save_path_loss = '../data/results/cnn/loss_fun/loss_fun.png'
    save_path_performance = '../data/results/cnn/performance'
    save_path_model = '../data/cnn_models'
    save_path_distribution = '../data/results/cnn/distribution/distribution.png'
    save_path_predictions = '../data/results/cnn/predictions/predictions.png'

    print(X_train_tensor.shape)
    #cross_validate_model(NN, X_train_tensor, y_train_tensor, train_indices, epochs=20,  n_splits=3, lr=5e-6)

    model = NN(X_train_tensor.shape).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.RMSprop(model.parameters(), lr=5e-6) # Adam
    trainer = ModelTrainer(model, criterion, optimizer, num_epochs=75, device=device)
    trainer.train(train_loader, test_loader)

    # Evaluate on the test set
    test_loss, test_outputs, test_targets, test_indices_eval = trainer.evaluate_model(test_loader)

    # Rescale the outputs and targets
    test_outputs_rescaled, test_targets_rescaled = rescale('../data/scaler/scaler.pkl', test_outputs,
                                                           test_targets)

    plot_predictions(test_targets, test_outputs, save_path=save_path_predictions)
    plot_distribution(test_targets_rescaled, test_outputs, save_path=save_path_distribution)

    print(f'Test Loss: {test_loss:.4f}')
    print('Test Targets:', train_indices, train_indices.shape)
    print('Test', test_outputs_rescaled, test_outputs_rescaled.shape)
    print('Test', test_targets_rescaled, test_targets_rescaled.shape)
    print("t", test_indices)
    #outfile = np.concatenate((
    #    test_indices_eval.reshape(-1, 1).astype(int), test_targets_rescaled, test_outputs_rescaled,
    #    test_targets, test_outputs), axis=1)
    colnames = ['idx', 'target', 'prediction', 'target_scaled', 'prediction_scaled']
    #print(outfile)


if __name__ == '__main__':
    main()
