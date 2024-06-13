from types import NoneType
from typing import Tuple
import numpy as np
from dataclasses import dataclass
import pandas as pd
import os

from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from pandas import DataFrame, Series
from src.util import match_df, remove_outliers, remove_outliers_iqr, fit_transform_df

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

    def pre_process_data(self, test_size: float = 0.3, z_value: float = 3.) -> None:
        """
        Function to pre-process the input data including:
        - Filtering columns
        - Adding new columns
        - Removing NaN and outliers
        - Test train split
        - Scaling the data

        :return: None
        """

        use_columns: list[str] = ['lot_area', 'building_area', 'commercial_area', 'residential_area',
                                  'office_area', 'retail_area', 'num_buildings', 'num_floors',
                                  'residential_units', 'total_units', 'lot_front', 'lot_depth',
                                  'building_front', 'building_depth', 'year_built']
        # Filter columns
        self.input_data = self.input_data[use_columns]

        # 'e_site_energy_use_norm_kbtu', 'e_total_site_energy_use_kbtu
        use_output_column: list[str] = ['e_site_energy_use_norm_kbtu']
        self.output_data = self.output_data[use_output_column]

        self.input_data = remove_outliers_iqr(self.input_data, q1=0.12, q2=0.88)
        self.output_data = remove_outliers_iqr(self.output_data, q1=0.12, q2=0.88)

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        # Check if all columns are of type numeric
        for column in self.input_data.columns:
            if self.input_data[column].dtype != 'float64':
                print(f'Column {column} is of type {self.input_data[column].dtype}')

        # Add new columns
        self.input_data.loc[:, 'lot_bldg_ratio'] = self.input_data['building_area'] / self.input_data['lot_area']
        self.input_data.loc[:, 'unit_area'] = self.input_data['building_area'] / self.input_data['total_units']

        # Remove NaN and outliers with z-score > 3
        self.input_data = self.input_data.replace([float('inf'), float('-inf')], pd.NA)
        self.output_data = self.output_data.replace([float('inf'), float('-inf')], pd.NA)

        self.input_data = self.input_data[~(self.input_data.isnull().any(axis=1))]
        self.output_data = self.output_data.loc[~(self.output_data.isnull().any(axis=1))]

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        self.input_data = remove_outliers(self.input_data, z_value, axis=1)
        self.output_data = remove_outliers(self.output_data, z_value, axis=0)  # axis=0 for output data

        self.input_data, self.output_data = match_df(self.input_data, self.output_data)

        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.input_data, self.output_data,
                                                                      test_size=test_size)

        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5)

        # Scale the data
        scaler = StandardScaler()

        self.X_train = fit_transform_df(self.X_train, scaler)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns=self.X_val.columns)

        self.y_train = fit_transform_df(self.y_train, scaler)
        self.y_test = pd.DataFrame(scaler.transform(self.y_test), columns=self.y_test.columns)
        self.y_val = pd.DataFrame(scaler.transform(self.y_val), columns=self.y_val.columns)

        self.input_data_scaled = pd.concat([self.X_train, self.X_test, self.X_val])
        self.output_data_scaled = pd.concat([self.y_train, self.y_test, self.y_val])

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

    def plot_correlation_matrix(self, save_path: str | None = None):
        """
        Function to plot the correlation matrix

        :return: None
        """

        if self.correlation is None:
            self.compute_correlation()

        plt.figure(figsize=(16, 13))
        ax = sns.heatmap(self.correlation, annot=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def plot_2d_correlation(self, x: str, save_path: str | None = None) -> None:
        """
        Function to plot the 2D correlation between two columns

        :param x: str: Name of the first column

        :return: None
        """

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
        ax.set_title(f'Linear fit between {x} and {self.output_data_scaled.columns[0]}. P-value: {p_value:.4f} {significance}')

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def plot_distribution(self, column: str, save_path: str | None = None) -> None:
        """
        Function to plot the distribution of a column

        :param column: str: Name of the column

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

        plt.show()

    def plot_boxplots(self, column: str, save_path: str | None = None) -> None:
        """
        Function to plot the boxplot of a column

        :param column:
        :param save_path:
        :return:
        """

        if self.input_data_scaled is None:
            raise ValueError('Data not loaded')

        if column not in self.input_data_scaled.columns:
            raise ValueError(f'{column} not in input data')

        fig, ax = plt.subplots(1, 1, figsize=(6, 10))
        sns.boxplot(
            y=self.input_data_scaled[column],
            ax=ax,
            flierprops={"marker": "x"},
            boxprops={"facecolor": "None"},
            linewidth=0.5)
        ax.set_title(f'Boxplot of {column}')

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


if __name__ == '__main__':
    data_model = DataModel()
    data_model.pre_process_data(test_size=0.2, z_value=3.)

    data_model.compute_correlation()
    data_model.plot_correlation_matrix(save_path=os.path.join(path_corr_matrix, 'correlation_matrix.png'))

    print(data_model.input_data_scaled)
    print(data_model.output_data_scaled)

    for column in data_model.input_data.columns:

        filename_2d_corr = f'{column}_2d_correlation.png'
        filename_distribution = f'{column}_distribution.png'
        filename_boxplot = f'{column}_boxplot.png'

        data_model.plot_2d_correlation(column, save_path=os.path.join(path_2d_corr, filename_2d_corr))
        data_model.plot_distribution(column, save_path=os.path.join(path_distribution, filename_distribution))
        data_model.plot_boxplots(column, save_path=os.path.join(path_boxplots, filename_boxplot))

    data_model.export_data(cnn_data_path)

