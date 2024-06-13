from types import NoneType
from typing import Tuple

import joblib
import numpy as np
from dataclasses import dataclass
import pandas as pd
import os

from scipy import stats
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from pandas import DataFrame, Series
from src.util import match_df, remove_outliers_zscore, remove_outliers_iqr, fit_transform_df

plt.style.use(['science', 'ieee'])

# GLOBAL VARIABLES
project_path: str = os.path.abspath(__file__)
root_path: str = os.path.dirname(os.path.dirname(project_path))

data_path: str = os.path.join(root_path, 'data')

_input_path: str = os.path.join(data_path, r'pre_processed\bef_input.csv')
_output_path: str = os.path.join(data_path, r'pre_processed\energy_output.csv')

path_corr_matrix: str = os.path.join(data_path, 'results', 'correlation_matrices')
path_2d_corr: str = os.path.join(data_path, 'results', 'correlations')
path_distribution: str = os.path.join(data_path, 'results', 'distributions')
path_boxplots: str = os.path.join(data_path, 'results', 'boxplots')

cnn_data_path: str = os.path.join(data_path, 'pre_processed', 'cnn_data')
scaler_data_path: str = os.path.join(data_path, 'scaler')

show_plots: bool = False


@dataclass
class DataModel:
    input_data: DataFrame | None = None
    output_data: DataFrame | None = None
    X_train: DataFrame | None = None
    X_test: DataFrame | None = None
    y_train: DataFrame | None = None
    y_test: DataFrame | None = None
    X_val: DataFrame | None = None
    y_val: DataFrame | None = None
    correlation: DataFrame | None = None
    input_data_scaled: DataFrame | None = None
    output_data_scaled: DataFrame | None = None

    def __post_init__(self):
        self.load_data(_input_path, _output_path)

    def load_data(self, path_in: str, path_out: str) -> None:
        self.input_data = pd.read_csv(path_in, index_col='bef_id')
        self.output_data = pd.read_csv(path_out, index_col='energy_id')

    def pre_process_data(self,
                         scaler: object = StandardScaler(),
                         test_size: float = 0.3,
                         z_value: float = 3.,
                         outlier_filter: str = 'zscore') -> None:
        """
        Function to pre-process the input data including:
        - Filtering columns
        - Adding new columns
        - Removing NaN and outliers
        - Test train split
        - Scaling the data

        :param test_size: (float) Size of the test data
        :param scaler: (object) Scaler object: MinMaxScaler or StandardScaler
        :param z_value: (float) Z-score threshold for outlier removal
        :param outlier_filter: (str) Method to remove outliers: 'zscore' or 'iqr'
        :return: None
        """
        """
        ['lot_area', 'building_area', 'commercial_area', 'residential_area', 'num_floors',
         'residential_units', 'total_units', 'lot_front', 'lot_depth', 'building_front',
         'building_depth', 'year_built', 'year_altered']"""

        use_columns: list[str] = ['building_area', 'residential_area', 'residential_units', 'total_units', 'year_built', 'year_altered']

        init_number_of_rows: int = len(self.input_data)

        # Filter columns
        self.input_data = self.input_data[use_columns]

        # 'e_site_energy_use_norm_kbtu', 'e_total_site_energy_use_kbtu'
        use_output_column: list[str] = ['e_total_site_energy_use_kbtu']
        self.output_data = self.output_data[use_output_column]

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        # Check if all columns are of type numeric
        for column in self.input_data.columns:
            if self.input_data[column].dtype != 'float64':
                print(f'Column {column} is of type {self.input_data[column].dtype}')

        # Add new columns
        # ----------------
        #
        # Ratio of building area to lot area
        if 'lot_area' in self.input_data.columns and 'building_area' in self.input_data.columns:
            self.input_data.loc[:, 'lot_bldg_ratio'] = self.input_data['building_area'] / self.input_data['lot_area']

        # Area per unit
        if 'total_units' in self.input_data.columns and 'building_area' in self.input_data.columns:
            self.input_data.loc[:, 'unit_area'] = self.input_data['building_area'] / self.input_data['total_units']

        # Ratio residential to commercial area
        if 'residential_area' in self.input_data.columns and 'commercial_area' in self.input_data.columns:
            self.input_data.loc[:, 'res_com_ratio'] = self.input_data['residential_area'] / self.input_data['commercial_area']

        # ----------------

        # Remove all rows where residential area is less than 0.75
        if 'residential_area' in self.input_data.columns:
            self.input_data = self.input_data[self.input_data['residential_area'] > 0.75]

        # Check how many rows are left
        print(f'Number of rows: {len(self.input_data)}')

        # Remove NaN and outliers with z-score > 3
        self.input_data = self.input_data.replace([float('inf'), float('-inf')], pd.NA)
        self.output_data = self.output_data.replace([float('inf'), float('-inf')], pd.NA)

        self.input_data = self.input_data[~(self.input_data.isnull().any(axis=1))]
        self.output_data = self.output_data.loc[~(self.output_data.isnull().any(axis=1))]

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        if outlier_filter == 'iqr':
            self.input_data = remove_outliers_iqr(self.input_data, q1=0.25, q2=0.75, axis=1)
            self.output_data = remove_outliers_iqr(self.output_data, q1=0.25, q2=0.75, axis=1)
        elif outlier_filter == 'zscore':
            self.input_data = remove_outliers_zscore(self.input_data, z_value, axis=1)
            self.output_data = remove_outliers_zscore(self.output_data, z_value, axis=0)  # axis=0 for output data
        else:
            raise ValueError(f'Invalid outlier filter: {outlier_filter}')

        print(f'Number of rows after outlier removal (input): {len(self.input_data)}')
        print(f'Number of rows after outlier removal (output): {len(self.output_data)}')

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.input_data, self.output_data,
                                                                      test_size=test_size)

        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5)

        self.X_train = fit_transform_df(self.X_train, scaler)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns=self.X_val.columns)

        joblib.dump(scaler, os.path.join(scaler_data_path, 'bef_scaler.pkl'))

        self.y_train = fit_transform_df(self.y_train, scaler)
        self.y_test = pd.DataFrame(scaler.transform(self.y_test), columns=self.y_test.columns)
        self.y_val = pd.DataFrame(scaler.transform(self.y_val), columns=self.y_val.columns)

        joblib.dump(scaler, os.path.join(scaler_data_path, 'energy_scaler.pkl'))

        self.input_data_scaled = pd.concat([self.X_train, self.X_test, self.X_val])
        self.output_data_scaled = pd.concat([self.y_train, self.y_test, self.y_val])

        print('---------------------------------')
        print('-- Data Pre-processing Summary --')
        print('--')
        print(f'-- Initial number of rows in the dataset: {init_number_of_rows}')
        print(f'-- Current number of rows in the dataset: {len(self.input_data_scaled)}')
        print('--')
        print(f'-- {len(self.input_data_scaled)/init_number_of_rows*100:.2f}% of the data remains')


    def export_data(self, path_out: str):
        """
        Function to export the data

        :param path_out: str: Path to the output file

        :return: None
        """

        # Define file names
        x_train_path = os.path.join(path_out, 'X_train.csv')
        x_test_path = os.path.join(path_out, 'X_test.csv')
        y_train_path = os.path.join(path_out, 'y_train.csv')
        y_test_path = os.path.join(path_out, 'y_test.csv')
        x_val_path = os.path.join(path_out, 'X_val.csv')
        y_val_path = os.path.join(path_out, 'y_val.csv')

        self.X_train.to_csv(x_train_path, index=True)
        self.X_test.to_csv(x_test_path, index=True)
        self.y_train.to_csv(y_train_path, index=True)
        self.y_test.to_csv(y_test_path, index=True)
        self.X_val.to_csv(x_val_path, index=True)
        self.y_val.to_csv(y_val_path, index=True)

    def compute_correlation(self) -> None:
        """
        Function to compute the correlation between the input and output data

        :return: None
       """

        combined_data = pd.concat([self.input_data, self.output_data], axis=1)
        self.correlation = combined_data.corr()
        self.p_values = self.correlation.apply(
            lambda x: [stats.pearsonr(x, self.correlation[col])[1] for col in self.correlation.columns], axis=0)

    def plot_correlation_matrix(self, save_path: str | None = None, show_plot: bool = False):
        """
        Function to plot the correlation matrix

        :param save_path: (str) Path to save the plot
        :param show_plot: (bool) Show the plot
        :return: None
        """

        if self.correlation is None:
            self.compute_correlation()

        fig, ax = plt.subplots(figsize=(20, 16))

        # Create a heatmap from the correlation matrix
        cax = ax.matshow(self.correlation, cmap='magma')

        # Create a colorbar for the heatmap
        fig.colorbar(cax)

        # Set the labels for the x-axis and y-axis
        ax.set_xticks(range(len(self.correlation.columns)))
        ax.set_yticks(range(len(self.correlation.columns)))
        ax.set_xticklabels(self.correlation.columns, rotation=45, ha='right', fontsize=20)
        ax.set_yticklabels(self.correlation.columns, fontsize=20)

        ax.xaxis.set_ticks_position('bottom')

        for i in range(self.correlation.shape[0]):
            for j in range(self.correlation.shape[1]):
                suffix: str = ' **' if self.p_values.iloc[i, j] <= 0.01 else ' *' if self.p_values.iloc[i, j] <= 0.05 else ''
                color: str = 'black' if np.abs(self.correlation.iloc[i, j]) > 0.7 else 'white'
                text = ax.text(j, i, str(np.around(self.correlation.iloc[i, j], decimals=2)) + suffix,
                               ha="center", va="center", color=color, fontsize=14)

        if save_path is not None:
            print("Saving correlation matrix to: ", save_path)
            plt.savefig(save_path)

        if show_plot:
            plt.show()

    def plot_2d_correlation(self, x: str, save_path: str | None = None, show_plot: bool = False) -> None:
        """
        Function to plot the 2D correlation between two columns

        :param x: (str) Name of the first column
        :param save_path: (str) Path to save the plot
        :param show_plot: (bool) Show the plot

        :return: None
        """
        plt.style.use([])

        if self.input_data_scaled is None:
            raise ValueError('Data not loaded')

        if x not in self.input_data_scaled.columns:
            raise ValueError(f'{x} not in input data')

        x_values = self.input_data_scaled.loc[:, x]
        y_values = self.output_data_scaled.iloc[:, 0]

        slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

        print(f"Slope: {slope}, Intercept: {intercept}, p-value: {p_value}")

        if p_value <= 0.01:
            significance = '**'
        elif p_value <= 0.05:
            significance = '*'
        else:
            significance = '[not significant]'

        line_values = intercept + slope * x_values

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.scatter(x_values, y_values, marker='^', alpha=0.8, s=1.5)
        ax.plot(x_values, line_values, color='red')
        ax.set_xlabel(x)
        ax.set_ylabel(self.output_data_scaled.columns[0])
        ax.set_title(
            f'Linear fit between {x} and {self.output_data_scaled.columns[0]}. P-value: {p_value:.4f} {significance}')

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot:
            plt.show()

    def plot_distribution(self, column: str, save_path: str | None = None, show_plot: bool = False) -> None:
        """
        Function to plot the distribution of a column

        :param column: (str) Name of the column
        :param save_path: (str) Path to save the plot
        :param show_plot: (bool) Show the plot

        :return: None
        """

        if self.input_data_scaled is None:
            raise ValueError('Data not loaded')

        if column not in self.input_data_scaled.columns:
            raise ValueError(f'{column} not in input data')

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.histplot(self.input_data_scaled[column], ax=ax)
        sns.kdeplot(self.input_data_scaled[column], color='red', ax=ax)
        ax.set_title(f'Distribution of {column}')

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot:
            plt.show()

    def plot_boxplots(self,
                      column: str,
                      save_path: str | None = None,
                      showfliers: bool = True,
                      show_plot: bool = False) -> None:
        """
        Function to plot the boxplot of a column

        :param column: (str) Name of the column
        :param save_path: (str) Path to save the plot
        :param showfliers: (bool) Show fliers in the boxplot
        :param show_plot: (bool) Show the plot
        :return: None
        """

        if self.input_data_scaled is None:
            raise ValueError('Data not loaded')

        if column not in self.input_data_scaled.columns:
            raise ValueError(f'{column} not in input data')

        fig, ax = plt.subplots(1, 1, figsize=(4, 10))

        b = sns.boxplot(
            y=self.input_data_scaled[column],
            ax=ax,
            flierprops={"marker": "x"},
            boxprops={"facecolor": "None"},
            linewidth=0.5,
            showfliers=showfliers)
        ax.set_title(f'Boxplot of {column.replace("_", " ")}', fontsize=16)

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot:
            plt.show()

    def plot_output_distribution(self, save_path: str | None = None) -> None:
        """
        Function to plot the distribution of the output data

        :param save_path: (str) Path to save the plot

        :return: None
        """

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.histplot(self.output_data_scaled, ax=ax)
        sns.kdeplot(self.output_data_scaled, color='red', ax=ax)
        ax.set_title('Distribution of Output Data')

        if save_path is not None:
            f_name: str = os.path.join(save_path, 'output_distribution.png')
            plt.savefig(f_name)
        plt.show()


if __name__ == '__main__':
    data_model = DataModel()
    data_model.pre_process_data(scaler=MinMaxScaler(), test_size=0.2, z_value=3., outlier_filter='iqr')

    data_model.compute_correlation()
    data_model.plot_correlation_matrix(save_path=os.path.join(path_corr_matrix, 'correlation_matrix.png'), show_plot=True)

    print(data_model.input_data_scaled)
    print(data_model.output_data_scaled)

    for column in data_model.input_data.columns:
        filename_2d_corr = f'{column}_2d_correlation.png'
        filename_distribution = f'{column}_distribution.png'
        filename_boxplot = f'{column}_boxplot.png'

        data_model.plot_2d_correlation(column, save_path=os.path.join(path_2d_corr, filename_2d_corr))
        data_model.plot_distribution(column, save_path=os.path.join(path_distribution, filename_distribution))
        data_model.plot_boxplots(column, save_path=os.path.join(path_boxplots, filename_boxplot), showfliers=False)

    data_model.export_data(cnn_data_path)
    data_model.plot_output_distribution(save_path=path_distribution)
