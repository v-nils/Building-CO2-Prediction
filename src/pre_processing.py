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

    def __post_init__(self):
        self.load_data(_input_path, _output_path)

    def load_data(self, path_in: str, path_out: str) -> None:
        self.input_data = pd.read_csv(path_in, index_col='bef_id')
        self.output_data = pd.read_csv(path_out, index_col='energy_id')

    def pre_process_data(self, test_size: float = 0.2, z_value: float = 3.) -> None:
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

        scaler = StandardScaler()
        z_scores_input = scaler.fit_transform(self.input_data)
        z_scores_output = scaler.fit_transform(self.output_data)

        self.input_data = self.input_data[(z_scores_input < z_value).all(axis=1)]
        self.output_data = self.output_data[(z_scores_output < z_value).all(axis=1)]

        drop_idx = list(set(self.input_data.index) ^ set(self.output_data.index))

        self.input_data = self.input_data.loc[~self.input_data.index.isin(drop_idx)]
        self.output_data = self.output_data.loc[~self.output_data.index.isin(drop_idx)]
        print('New shape of input data:', self.input_data.shape)

        assert (self.input_data.shape[0] == self.output_data.shape[0])

        # Test train split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.input_data,
                                                                                self.output_data, test_size=test_size)

        # Scale the data
        scaler = MinMaxScaler()
        z_scores_input = scaler.fit_transform(self.X_train)
        z_scores_output = scaler.fit_transform(self.y_train)

        self.input_data_scaled = pd.DataFrame(z_scores_input, columns=self.input_data.columns)
        self.output_data_scaled = pd.DataFrame(z_scores_output, columns=self.output_data.columns)

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.y_train = scaler.fit_transform(self.y_train)
        self.y_test = scaler.transform(self.y_test)

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
        else:
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
        ax.scatter(x_values, y_values)
        ax.plot(x_values, line_values, color='red')
        ax.set_xlabel(x)
        ax.set_ylabel(self.output_data_scaled.columns[0])
        ax.set_title(f'Linear fit between {x} and {self.output_data_scaled.columns[0]}. P-value: {p_value:.4f} {significance}')

        if save_path is not None:
            plt.savefig(save_path)
        else:
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

        else:
            plt.show()


if __name__ == '__main__':
    data_model = DataModel()
    data_model.pre_process_data(test_size=0.2, z_value=1.7)

    data_model.compute_correlation()
    data_model.plot_correlation_matrix()#save_path=os.path.join(path_corr_matrix, 'correlation_matrix.png'))

    print(data_model.input_data_scaled.head())
    print(data_model.output_data_scaled.head())

    for column in data_model.input_data.columns:

        filename_2d_corr = f'{column}_2d_correlation.png'
        filename_distribution = f'{column}_distribution.png'

        data_model.plot_2d_correlation(column)#, save_path=os.path.join(path_2d_corr, filename_2d_corr))
        data_model.plot_distribution(column)# , save_path=os.path.join(path_distribution, filename_distribution))

