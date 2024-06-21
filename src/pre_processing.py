import copy
from types import NoneType
from typing import Tuple

import joblib
import numpy as np
from dataclasses import dataclass
import pandas as pd
import os
from sklearn.linear_model import Lasso, Ridge
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from pandas import DataFrame, Series
from sympy.core.numbers import NegativeOne

from src.util import match_df, remove_outliers_zscore, remove_outliers_iqr, fit_transform_df

plt.style.use(['science', 'ieee'])

# GLOBAL VARIABLES
project_path: str = os.path.abspath(__file__)
root_path: str = os.path.dirname(os.path.dirname(project_path))

use_output_column: list[str] = ['e_total_site_energy_use_kbtu']
output_label = 'Total energy consumption' if use_output_column == 'e_total_site_energy_use_kbtu' else 'Normalized energy consumption'

bef_dict: dict[str, str] = {
    "lot_area": "Lot area",
    "building_area": "Building area",
    "commercial_area": "Commercial area",
    "residential_area": "Residential area",
    "num_floors": "Number of floors",
    "residential_units": "Residential units",
    "total_units": "Total units",
    "lot_front": "Lot frontage",
    "lot_depth": "Lot depth",
    "building_front": "Building frontage",
    "building_depth": "Building depth",
    "year_built": "Year built",
    "year_altered": "Year altered",
    "Z_Min": "Min elevation",
    "Z_Max": "Max elevation",
    "Z_Mean": "Mean elevation",
    "SArea": "Surface area",
    "Volume": "Volume",
    "floor_area_ratio": "Floor area ratio",
    "lot_bldg_ratio": "Lot building ratio",
    "unit_area": "Unit area",
    "res_ratio": "Residential ratio",
    "com_ratio": "Commercial ratio",
    "sarea_volume_ratio": "Surface area to volume ratio",
    "e_site_energy_use_norm_kbtu": "Normalized energy consumption",
    "e_total_site_energy_use_kbtu": "Total energy consumption"
}

data_path: str = os.path.join(root_path, 'data')

_input_path: str = os.path.join(data_path, r'pre_processed\bef_input.csv')
_output_path: str = os.path.join(data_path, r'pre_processed\energy_output.csv')
_data_path: str = os.path.join(data_path, r'pre_processed\bef_energy_data.csv')

path_corr_matrix: str = os.path.join(data_path, 'results', 'correlation_matrices')
path_2d_corr: str = os.path.join(data_path, 'results', 'correlations')
path_distribution: str = os.path.join(data_path, 'results', 'distributions')
path_r2: str = os.path.join(data_path, 'results', 'r2_values')
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
    correlation: DataFrame | None = None
    input_data_scaled: DataFrame | None = None
    output_data_scaled: DataFrame | None = None
    p_value: DataFrame | None = None
    r_2: dict[str, float | None ] | None = None

    def __len__(self) -> int:
        return len(self.input_data)

    def __post_init__(self):

        data: pd.DataFrame = pd.read_csv(_data_path, delimiter=',')

        nan_per_column = data.isna().sum()
        self.input_data = data.drop(columns=['e_site_energy_use_norm_kbtu', 'e_total_site_energy_use_kbtu'])
        self.output_data = data[['e_total_site_energy_use_kbtu']]
        print(self.input_data.head())

    def load_data(self, path_in: str, path_out: str) -> None:
        self.input_data = pd.read_csv(path_in, index_col='bef_id')
        self.output_data = pd.read_csv(path_out, index_col='energy_id')

    def pre_process_data(self,
                         scaler: object = StandardScaler(),
                         test_size: float = 0.3,
                         z_value: float = 3.,
                         outlier_filter: str = 'zscore',
                         columns: list[str] | str = 'all') -> None:
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
        :param columns: (list) List of columns to keep
        :return: None
        """
        """
        ['lot_area', 'building_area', 'commercial_area', 'residential_area', 'num_floors',
         'residential_units', 'total_units', 'lot_front', 'lot_depth', 'building_front',
         'building_depth', 'year_built', 'year_altered', 'Z_Min', 'Z_Max', 'Z_Mean', 'SArea', 'Volume', 'floor_area_ratio', 
         'lot_bldg_ratio', 'unit_area', 'res_ratio', 'com_ratio', 'sarea_volume_ratio', 'e_site_energy_use_norm_kbtu', 'e_total_site_energy_use_kbtu']"""

        """
        'building_area', 'commercial_area', 'residential_area', 'residential_units',
                                  'total_units', 'year_built',
                                  'year_altered', 'num_floors', 'Z_Min', 'Z_Max', 'Z_Mean', 'SArea', 'Volume'
        """

        init_number_of_rows: int = len(self.input_data)

        # Filter columns
        if columns != 'all':
            self.input_data = self.input_data[columns]
        else:
            self.input_data = self.input_data[
                ['lot_area', 'building_area', 'commercial_area',
                 'residential_area', 'num_floors', 'residential_units',
                 'total_units', 'lot_front', 'lot_depth',
                 'building_front', 'building_depth', 'year_built',
                 'year_altered', 'Z_Min', 'Z_Max',
                 'Z_Mean', 'SArea', 'Volume',
                 'floor_area_ratio', 'lot_bldg_ratio', 'unit_area',
                 'res_ratio', 'com_ratio', 'sarea_volume_ratio']]

        # 'e_site_energy_use_norm_kbtu', 'e_total_site_energy_use_kbtu'
        self.output_data = self.output_data[use_output_column]

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        # Check if all columns are of type numeric
        for column in self.input_data.columns:
            if self.input_data[column].dtype != 'float64':
                print(f'Column {column} is of type {self.input_data[column].dtype}')

        self.input_data = self.input_data.replace([float('inf'), float('-inf')], pd.NA)
        self.output_data = self.output_data.replace([float('inf'), float('-inf')], pd.NA)

        self.input_data = self.input_data[~(self.input_data.isnull().any(axis=1))]
        self.output_data = self.output_data.loc[~(self.output_data.isnull().any(axis=1))]

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        if outlier_filter == 'iqr':
            self.input_data = remove_outliers_iqr(self.input_data, q1=0.3, q2=0.7, axis=1)
            self.output_data = remove_outliers_iqr(self.output_data, q1=0.25, q2=0.75, axis=1)
        elif outlier_filter == 'zscore':
            self.input_data = remove_outliers_zscore(self.input_data, z_value, axis=1)
            self.output_data = remove_outliers_zscore(self.output_data, z_value, axis=0)  # axis=0 for output data
        else:
            raise ValueError(f'Invalid outlier filter: {outlier_filter}')

        print(f'Number of rows after outlier removal (input): {len(self.input_data)}')
        print(f'Number of rows after outlier removal (output): {len(self.output_data)}')

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.input_data, self.output_data,
                                                                                test_size=test_size)

        self.X_train = fit_transform_df(self.X_train, scaler)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index)

        joblib.dump(scaler, os.path.join(scaler_data_path, 'bef_scaler.pkl'))

        self.y_train = fit_transform_df(self.y_train, scaler)
        self.y_test = pd.DataFrame(scaler.transform(self.y_test), columns=self.y_test.columns, index=self.y_test.index)

        joblib.dump(scaler, os.path.join(scaler_data_path, 'energy_scaler.pkl'))

        self.input_data_scaled = pd.concat([self.X_train, self.X_test])
        self.output_data_scaled = pd.concat([self.y_train, self.y_test])

        print('---------------------------------')
        print('-- Data Pre-processing Summary --')
        print('--')
        print(f'-- Initial number of rows in the dataset: {init_number_of_rows}')
        print(f'-- Current number of rows in the dataset: {len(self.input_data_scaled)}')
        print('--')
        print(f'-- Number of rows in the training set: {len(self.X_train)}')
        print(f'-- Number of rows in the test set: {len(self.X_test)}')
        print(f'-- {len(self.input_data_scaled) / init_number_of_rows * 100:.2f}% of the data remains')

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
        x_all_path = os.path.join(path_out, 'X_all.csv')
        y_all_path = os.path.join(path_out, 'y_all.csv')

        self.X_train.to_csv(x_train_path, index=True)
        self.X_test.to_csv(x_test_path, index=True)
        self.y_train.to_csv(y_train_path, index=True)
        self.y_test.to_csv(y_test_path, index=True)
        self.input_data_scaled.to_csv(x_all_path, index=True)
        self.output_data_scaled.to_csv(y_all_path, index=True)

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

        fig, ax = plt.subplots(figsize=(28, 24))

        # Create a heatmap from the correlation matrix
        cax = ax.matshow(self.correlation, cmap='magma')

        # Create a colorbar for the heatmap
        fig.colorbar(cax)

        tick_labels = [bef_dict[col] for col in self.correlation.columns]

        # Set the labels for the x-axis and y-axis
        ax.set_xticks(range(len(self.correlation.columns)))
        ax.set_yticks(range(len(self.correlation.columns)))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=20)
        ax.set_yticklabels(tick_labels, fontsize=20)

        ax.xaxis.set_ticks_position('bottom')

        for i in range(self.correlation.shape[0]):
            for j in range(self.correlation.shape[1]):
                suffix: str = ' **' if self.p_values.iloc[i, j] <= 0.01 else ' *' if self.p_values.iloc[
                                                                                         i, j] <= 0.05 else ''
                color: str = 'black' if np.abs(self.correlation.iloc[i, j]) > 0.7 else 'white'
                text = ax.text(j, i, str(np.around(self.correlation.iloc[i, j], decimals=2)) + suffix,
                               ha="center", va="center", color=color, fontsize=14)

        if save_path is not None:
            print("Saving correlation matrix to: ", save_path)
            plt.savefig(save_path)

        if show_plot is True:
            plt.show()

    def compute_2d_correlation(self, x: str, save_path: str | None = None, show_plot: bool = False) -> None:
        """
        Function to plot the 2D correlation between two columns

        :param x: (str) Name of the first column
        :param save_path: (str) Path to save the plot
        :param show_plot: (bool) Show the plot

        :return: None
        """
        plt.style.use([])

        if self.r_2 is None:
            self.r_2 = {}

        if self.input_data_scaled is None:
            raise ValueError('Data not loaded')

        if x not in self.input_data_scaled.columns:
            raise ValueError(f'{x} not in input data')

        x_values = self.input_data_scaled.loc[:, x].to_numpy().reshape(-1, 1)
        y_values = self.output_data_scaled.iloc[:, 0].to_numpy().reshape(-1, 1)

        print("-----------------------------")
        print("-- START Lasso regression --")
        print(f"-- Feature: {x}")
        print("-----------------------------")

        lasso = Ridge(alpha=1e-3, max_iter=30_000, solver='svd')
        lasso.fit(x_values, y_values)
        y_pred = lasso.predict(x_values)

        # Get the coefficients
        slope = lasso.coef_[0][0]
        intercept = lasso.intercept_

        print(f"Slope: {slope}")

        r_2 = r2_score(y_values, y_pred)

        self.r_2[x] = r_2

        fig, ax = plt.subplots(1, 1, figsize=(11, 9))
        ax.scatter(x_values, y_values, marker='x', color='black', alpha=0.6, s=15, label='Data points')

        ax.plot(x_values, y_pred, color='red', linewidth=1.5, linestyle='--',
                label='Ridge (L1) regression - Slope: {:.2f}'.format(np.round(slope, 2)))

        ax.set_xlabel(bef_dict[x], fontsize=20)
        ax.set_ylabel(output_label, fontsize=20)
        ax.set_title('${}^{}$={}'.format('R', 2, np.round(r_2, 3)), fontsize=24)

        ax.tick_params(axis='both', colors='black', labelsize=12)
        plt.legend(fontsize=16)

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot is True:
            plt.show()

        # close the plot
        plt.close()

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

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

        # Plot the histogram
        sns.histplot(self.input_data_scaled[column], ax=ax1, kde=False, element="bars", edgecolor=None, color='gray',
                     label='Bins')
        ax1.set_xlabel(bef_dict[column], fontsize=20)
        ax1.set_ylabel(f'Frequency', fontsize=20)
        ax1.tick_params(axis='both', colors='black', labelsize=12)

        # Create a secondary y-axis for the KDE plot
        ax2 = ax1.twinx()
        sns.kdeplot(self.input_data_scaled[column], color='black', linestyle='-.', linewidth=1.5, ax=ax2,
                    label='Kernel Density Estimate')
        ax2.set_ylabel('Density', fontsize=20)
        ax2.tick_params(axis='y', colors='black', labelsize=12)
        plt.legend(fontsize=16)

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot is True:
            plt.show()

        plt.close()

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
            linewidth=0.8,
            showfliers=showfliers)

        ax.set_ylabel(bef_dict[column], fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot is True:
            plt.show()

        plt.close()

    def plot_output_distribution(self, save_path: str | None = None, show_plot: bool = False) -> None:
        """
        Function to plot the distribution of the output data

        :param save_path: (str) Path to save the plot
        :param show_plot: (bool) Show the plot

        :return: None
        """

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

        # Plot the histogram
        sns.histplot(self.output_data_scaled, ax=ax1, kde=False, element="bars", edgecolor=None, color='gray',
                     label='Bins')
        ax1.set_xlabel(output_label, fontsize=20)
        ax1.set_ylabel('Frequency', fontsize=20)
        ax1.tick_params(axis='both', colors='black', labelsize=12)

        # Create a secondary y-axis for the KDE plot
        ax2 = ax1.twinx()
        sns.kdeplot(self.output_data_scaled, color='red', linewidth=1.5, linestyle='-.', ax=ax2,
                    label='Kernel Density Estimate')
        ax2.set_ylabel('Density', fontsize=20)
        ax2.yaxis.label.set_color('black')
        ax2.tick_params(axis='y', colors='black', labelsize=12)
        plt.legend(fontsize=16)

        if save_path is not None:
            f_name: str = os.path.join(save_path, 'output_distribution.png')
            plt.savefig(f_name)

        if show_plot is True:
            plt.show()

        plt.close()

    def filter_columns(self, threshold: float = 0.3, model_params: dict | None = None) -> None:
        """
        Function to filter columns

        :return:
        """

        # Get columns where r_2 > 0.3
        columns = [key for key, value in self.r_2.items() if value > threshold]

        self.__post_init__()
        self.pre_process_data(**model_params, columns=columns)

    def plot_r2(self, save_path: str | None = None, show_plot: bool = False) -> None:
        """
        Function to plot the R^2 values

        :param save_path: (str) Path to save the plot
        :param show_plot: (bool) Show the plot

        :return: None
        """

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        x_values = list(self.r_2.keys())
        y_values = list(self.r_2.values())

        # Sort the values
        x_values, y_values = zip(*sorted(zip(x_values, y_values), key=lambda x: x[1], reverse=True))

        x_labels = [bef_dict[col] for col in x_values]

        bars = ax.bar(x_values, y_values, color='gray', edgecolor='black', linewidth=0.8)

        ax.set_ylabel('R$^2$ magnitude [0, 1]', fontsize=20)
        ax.set_title('R$^2$ values for the features', fontsize=24)

        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=20)
        ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1.5, label='Threshold: 0.3')

        ax.tick_params(axis='both', colors='black', labelsize=12)

        # Add the actual R^2 values inside the bars or on top if too small
        for bar, value in zip(bars, y_values):
            height = bar.get_height()
            if height < 0.05:  # Adjust this threshold as needed
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{value:.2f}', ha='center', va='bottom',
                        fontsize=12)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{value:.2f}', ha='center', va='center',
                        fontsize=12, color='white')

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot is True:
            plt.show()

        plt.close()


if __name__ == '__main__':

    # Define the model
    data_model = DataModel()

    # ----------------------------------------------
    # --  Explorative Data Analysis (EDA)         --
    # ----------------------------------------------

    pre_processing_params = {'scaler': MinMaxScaler(), 'test_size': 0.2, 'z_value': 3, 'outlier_filter': 'iqr'}
    data_model.pre_process_data(**pre_processing_params, columns='all')

    data_model.compute_correlation()
    data_model.plot_correlation_matrix(save_path=os.path.join(path_corr_matrix, 'correlation_matrix.png'),
                                       show_plot=True)

    for column in data_model.input_data.columns:
        filename_2d_corr = f'{column}_2d_correlation.png'
        filename_distribution = f'{column}_distribution.png'
        filename_boxplot = f'{column}_boxplot.png'

        data_model.compute_2d_correlation(column, save_path=os.path.join(path_2d_corr, filename_2d_corr),
                                          show_plot=show_plots)

        data_model.plot_distribution(column, save_path=os.path.join(path_distribution, filename_distribution),
                                     show_plot=show_plots)

        data_model.plot_boxplots(column, save_path=os.path.join(path_boxplots, filename_boxplot), show_plot=show_plots,
                                 showfliers=False)

    data_model.plot_r2(save_path=os.path.join(path_r2, 'r2.png'), show_plot=show_plots)
    # ----------------------------------------------
    # --  Select the relevant columns             --
    # ----------------------------------------------

    data_model.filter_columns(threshold=0.1, model_params=pre_processing_params)

    # Repeat the pre-processing step with the selected columns

    data_model.export_data(cnn_data_path)
    data_model.plot_output_distribution(save_path=path_distribution)
