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
    'train': ('../data/pre_processed/cnn_data/X_train.csv', '../data/pre_processed/cnn_data/y_train.csv'),
    'test': ('../data/pre_processed/cnn_data/X_test.csv', '../data/pre_processed/cnn_data/y_test.csv'),
    'all': ('../data/pre_processed/cnn_data/X_all.csv', '../data/pre_processed/cnn_data/y_all.csv')
}

final_results = '../data/in_out/final_results.csv'


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
    return X_tensor, y_tensor, X.index


# Load the data
X_train_tensor, y_train_tensor, train_indices = load_and_prepare_data(*file_paths['train'])
X_test_tensor, y_test_tensor, test_indices = load_and_prepare_data(*file_paths['test'])
X_all_tensor, y_all_tensor, all_indices = load_and_prepare_data(*file_paths['test'])

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


class CNN(nn.Module):
    def __init__(self):
        """
        Initialize the CNN model.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(352, 1)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.xavier_uniform_(self.fc1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
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

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path_performance: str,
              save_path_model: str, **loss_kwargs):
        """
        Train the model.

        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param save_path_loss: Path to save the loss plot.
        :param save_path_performance: Path to save the performance metrics.
        :param save_path_model: Path to save the trained model.
        """

        ax_t = loss_kwargs.get('ax_t', None)
        ax_v = loss_kwargs.get('ax_v', None)
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

        if ax_t is not None and ax_v is not None:
            self.plot_loss(ax_t, ax_v, self.train_losses, self.val_losses,
                           loss_kwargs.get('num_plot'))

        if save_path_model:
            self.save_model(save_path_model)

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


    def plot_loss(self, ax_t, ax_v, train_losses: list, val_losses: list, num_plot: int = 0):
        """
        Plot the training and validation loss.

        :param loss_axis: Axis to plot the loss.
        :param train_losses: List of training losses.
        :param val_losses: List of validation losses.
        """

        color = iter(cm.YlGn(np.linspace(0, 1, 20)))

        for i in range(num_plot + 1):
            train_col = next(color)


        ax_t.plot(train_losses, label=r'Training {}'.format(num_plot + 1), linewidth=1.5, linestyle='-.',
                   markersize=3, marker='^',
                   color=train_col)

        ax_v.plot(val_losses, label=r'Validation {}'.format(num_plot + 1), linewidth=1.5, linestyle='-.',
                   markersize=3, marker='o',
                   color=train_col)

        ax_t.set_xticklabels([])
        ax_v.set_xticklabels([i if i % 10 == 0 else '' for i in range(1, self.num_epochs + 1)])
        ax_t.tick_params(which='minor', length=0, color='r')
        ax_t.tick_params(which='major', length=0, color='r')
        ax_v.tick_params(which='minor', length=0, color='r')

        ax_t.set_ylabel('Loss [Training]', fontsize=16)

        ax_v.set_xlabel('Epochs', fontsize=16)
        ax_v.set_ylabel('Loss [Validation]', fontsize=16)


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


def plot_predictions(targets: np.ndarray, outputs: np.ndarray, indices: np.ndarray, save_path: str = None):
    """
    Plot the targets and outputs.

    :param targets: Actual target values.
    :param outputs: Predicted output values.
    :param indices: Original indices of the samples.
    :param save_path: Path to save the plot.
    """
    min_val = min(np.min(targets), np.min(outputs))
    max_val = max(np.max(targets), np.max(outputs))

    rmse = np.sqrt(mean_squared_error(targets, outputs))
    r2 = 1 - (np.sum((targets - outputs) ** 2) / np.sum((targets - np.mean(targets)) ** 2))

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    ax.scatter(targets, outputs, label='Outputs', color='red', marker='x', s=20, alpha=0.6)
    ax.set_xlabel('Target', fontsize=20)
    ax.set_ylabel('Prediction', fontsize=20)
    ax.set_title(r'${}^{}={}$'.format('R', 2, np.round(r2, 3)), fontsize=24)
    ax.tick_params(axis='both', colors='black', labelsize=12)
    ax.legend(fontsize=16)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    if save_path:
        plt.savefig(save_path)


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
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

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


def cross_validate_model(model_class: type, X_tensor: torch.Tensor, y_tensor: torch.Tensor, indices: pd.Index,
                         n_splits: int = 5, epochs: int = 25, lr: float = 0.0001, **kwargs):
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
    loss_fig, loss_ax = plt.subplots(2,1, figsize=(15, 12))

    round = 0
    for train_index, val_index in kf.split(X_tensor):
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]
        train_indices, val_indices = indices[train_index], indices[val_index]

        train_dataset = DataSet(X_train, y_train, train_indices)
        val_dataset = DataSet(X_val, y_val, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = model_class().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = ModelTrainer(model, criterion, optimizer, epochs, device)
        trainer.train(train_loader, val_loader, kwargs.get('save_path_performance'),
                      kwargs.get('save_path_model'), ax_t=loss_ax[0], ax_v=loss_ax[1], num_plot=round)

        val_loss, _, _, _ = trainer.evaluate_model(val_loader)
        mse_scores.append(val_loss)
        round += 1

    plt.xticks(np.arange(1, epochs, 1))
    plt.xlim(-0.1, epochs - 0.9)
    plt.legend()
    plt.savefig(kwargs.get('save_path_loss'))
    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f'Cross-Validated MSE: {avg_mse:.4f} ± {std_mse:.4f}')

    return trainer


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
    test_dataset = DataSet(X_test_tensor, y_test_tensor, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train the model
    save_path_loss = '../data/results/cnn/loss_fun/loss_fun.png'
    save_path_performance = '../data/results/cnn/performance'
    save_path_model = '../data/cnn_models'
    save_path_distribution = '../data/results/cnn/distribution/distribution.png'
    save_path_predictions = '../data/results/cnn/predictions/predictions.png'

    trainer = cross_validate_model(CNN, X_train_tensor, y_train_tensor, train_indices, epochs=20, n_splits=3, lr=5e-6,
                                   save_path_model=save_path_model,
                                   save_path_loss=save_path_loss, save_path_performance=save_path_performance)

    # Evaluate on the test set
    test_loss, test_outputs, test_targets, test_indices_eval = trainer.evaluate_model(test_loader)

    # Rescale the outputs and targets
    test_outputs_rescaled, test_targets_rescaled = rescale('../data/scaler/energy_scaler.pkl', test_outputs,
                                                           test_targets)

    plot_predictions(test_targets, test_outputs, test_indices_eval, save_path=save_path_predictions)
    plot_distribution(test_targets_rescaled, test_outputs, save_path=save_path_distribution)

    print(f'Test Loss: {test_loss:.4f}')
    print('Test Targets:', train_indices, train_indices.shape)
    print('Test', test_outputs_rescaled, test_outputs_rescaled.shape)
    print('Test', test_targets_rescaled, test_targets_rescaled.shape)
    print("t", test_indices)
    outfile = np.concatenate((
        test_indices_eval.reshape(-1, 1).astype(int), test_targets_rescaled, test_outputs_rescaled,
        test_targets, test_outputs), axis=1)
    colnames = ['idx', 'target', 'prediction', 'target_scaled', 'prediction_scaled']
    print(outfile)

    # Save the final results
    pd.DataFrame(outfile, columns=colnames).to_csv(final_results, index=False)

    # Compute over all data

    all_dataset = DataSet(X_all_tensor, y_all_tensor, train_indices)
    all_loader = DataLoader(all_dataset, batch_size=32, shuffle=False)

    print("All Targets:", all_dataset.indices.shape)

    all_loss, all_outputs, all_targets, all_idxs = trainer.evaluate_model(all_loader)

    print("All", all_outputs, all_outputs.shape)
    print("All", all_targets, all_targets.shape)
    all_outputs_rescaled, all_targets_rescaled = rescale('../data/scaler/energy_scaler.pkl', all_outputs, all_targets)
    plot_predictions(all_targets, all_outputs, all_idxs, save_path='../data/results/cnn/predictions/predictions_all.png')
    plot_distribution(all_targets_rescaled, all_outputs, save_path='../data/results/cnn/distribution/distribution_all.png')
    print(f'All Loss: {all_loss:.4f}')

    outfile = np.concatenate((
        all_idxs.reshape(-1, 1).astype(int), all_targets_rescaled, all_outputs_rescaled,
        all_targets, all_outputs), axis=1)
    colnames = ['idx', 'target', 'prediction', 'target_scaled', 'prediction_scaled']
    print(outfile)
    pd.DataFrame(outfile, columns=colnames).to_csv('../data/in_out/final_results_all.csv', index=False)


if __name__ == '__main__':
    main()
