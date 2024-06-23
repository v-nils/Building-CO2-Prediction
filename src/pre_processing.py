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

from src.util import match_df, remove_outliers_zscore, remove_outliers_iqr, fit_transform_df, transform_df
import warnings

warnings.filterwarnings("ignore")

plt.style.use(['science', 'ieee'])

# GLOBAL VARIABLES
project_path: str = os.path.abspath(__file__)
root_path: str = os.path.dirname(os.path.dirname(project_path))

use_output_column: list[str] = ['e_total_site_energy_use_kbtu']
output_label = 'Total energy consumption' if use_output_column == 'e_total_site_energy_use_kbtu' else 'Normalized energy consumption'

bef_dict = {
    "bbl": "BBL",
    "lotarea": "Lot area",
    "bldgarea": "Building area",
    "comarea": "Commercial area",
    "resarea": "Residential area",
    "officearea": "Office area",
    "retailarea": "Retail area",
    "garagearea": "Garage area",
    "strgearea": "Storage area",
    "factryarea": "Factory area",
    "otherarea": "Other area",
    "numbldgs": "Number of buildings",
    "numfloors": "Number of floors",
    "unitstotal": "Total units",
    "lotfront": "Lot frontage",
    "lotdepth": "Lot depth",
    "bldgfront": "Building frontage",
    "bldgdepth": "Building depth",
    "lotbldgratio": "Lot/building ratio",
    "floorarearatio": "Floor area ratio",
    "unitarea": "Unit area",
    "resratio": "Residential ratio",
    "comratio": "Commercial ratio",
    "sareavolumeratio": "Surface/volume ratio",
    "sareabldgratio": "Surface/building ratio",
    "sarealotratio": "Surface/lot ratio",
    "volumebldgratio": "Volume/building ratio",
    "volumelotratio": "Volume/lot ratio",
    "assessland": "Land assessment",
    "assesstot": "Total assessment",
    "proxcode": "Proximity code",
    "yearbuilt": "Year built",
    "yearaltered": "Year altered",
    "irrlotcode": "Irregular lot code",
    "lottype": "Lot type",
    "bsmtcode": "Basement code",
    "builtfar": "Built FAR",
    "residfar": "Residential FAR",
    "commfar": "Commercial FAR",
    "facilfar": "Facility FAR",
    "shape_area": "Shape area",
    "z_min": "Min elevation",
    "z_max": "Max elevation",
    "z_mean": "Mean elevation",
    "sarea": "Surface area",
    "volume": "Volume",
    "weather_normalized_site_energy_use_kbtu": "Weather normalized site energy use (kBTU)",
    "weather_normalized_site_eui_kbtu_ft": "Weather normalized site EUI (kBTU/ft$^2$)",
    "total_ghg_emissions_metric_tons_co2e": "Total GHG emissions (metric tons CO$_2$e)",
    "total_ghg_emissions_intensity_kgco2e_ft": "Total GHG emissions intensity (kgCO$_2$e/ft$^2$)"
}


data_path: str = os.path.join(root_path, 'data')

_input_path: str = os.path.join(data_path, r'pre_processed\bef_input.csv')
_output_path: str = os.path.join(data_path, r'pre_processed\energy_output.csv')
_data_path: str = os.path.join(data_path, r'pre_processed\full_ds.csv')

path_corr_matrix: str = os.path.join(data_path, 'results', 'correlation_matrices')
path_2d_corr: str = os.path.join(data_path, 'results', 'correlations')
path_distribution: str = os.path.join(data_path, 'results', 'distributions')
path_r2: str = os.path.join(data_path, 'results', 'r2_values')
path_boxplots: str = os.path.join(data_path, 'results', 'boxplots')

cnn_data_path: str = os.path.join(data_path, 'pre_processed', 'nn_data')
scaler_data_path: str = os.path.join(data_path, 'scaler')

show_plots: bool = False


@dataclass
class DataModel:
    input_data: DataFrame | None = None
    processing_data: DataFrame | None = None
    output_data: DataFrame | None = None
    train: DataFrame | None = None
    test: DataFrame | None = None
    correlation: DataFrame | None = None
    data_scaled: DataFrame | None = None
    p_value: DataFrame | None = None
    r_2: dict[str, float | None] | None = None

    def __len__(self) -> int:
        return len(self.input_data)

    def __post_init__(self):

        self.df: pd.DataFrame = pd.read_csv(_data_path, delimiter=',', index_col=0)

    def load_data(self, path_in: str, path_out: str) -> None:
        self.input_data = pd.read_csv(path_in, index_col='bbl')
        self.output_data = pd.read_csv(path_out, index_col='energy_id')

    def pre_process_data(self,
                         scaler: object = StandardScaler(),
                         test_size: float = 0.3,
                         z_value: float = 3.,
                         outlier_filter: str = 'zscore',
                         input_columns: list[str] | str = 'all',
                         output_columns: str = 'total_ghg_emissions_metric_tons_co2e') -> None:
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
        :param input_columns: (list) List of columns to keep
        :param output_columns: (str) Name of the output column
        :return: None
        """

        init_number_of_rows: int = len(self.df)

        if input_columns == 'all':
            self.processing_data = pd.concat([self.df.iloc[:, :-4], self.df[[output_columns]]], axis=1)
        else:
            self.processing_data = pd.concat([self.df[input_columns], self.df[[output_columns]]], axis=1)

        # Check if all columns are of type numeric
        for column in self.processing_data.columns:
            if self.processing_data[column].dtype != 'float64' and self.processing_data[column].dtype != 'int64':
                print(f'Column {column} is of type {self.processing_data[column].dtype}')

        self.processing_data = self.processing_data.replace([np.inf, -np.inf], np.nan)

        self.processing_data = self.processing_data[~(self.df.isnull().any(axis=1))]

        if outlier_filter == 'iqr':
            self.processing_data = remove_outliers_iqr(self.processing_data, q1=0.1, q2=0.9, axis=1)
        elif outlier_filter == 'zscore':
            self.processing_data = remove_outliers_zscore(self.processing_data, z_value, axis=1)
        else:
            raise ValueError(f'Invalid outlier filter: {outlier_filter}')

        print(f'Number of rows after outlier removal: {len(self.processing_data)}')

        self.train, self.test = train_test_split(self.processing_data, test_size=test_size)

        # DropNaN
        self.train = self.train.dropna()
        self.test = self.test.dropna()

        self.train = fit_transform_df(self.train, scaler)
        self.test = transform_df(self.test, scaler)

        joblib.dump(scaler, os.path.join(scaler_data_path, 'scaler.pkl'))

        self.data_scaled = pd.concat([self.train, self.test], axis=0)

        print('---------------------------------')
        print('-- Data Pre-processing Summary --')
        print('--')
        print(f'-- Initial number of rows in the dataset: {init_number_of_rows}')
        print(f'-- Current number of rows in the dataset: {len(self.data_scaled)}')
        print('--')
        print(f'-- Number of rows in the training set: {len(self.train)}')
        print(f'-- Number of rows in the test set: {len(self.test)}')
        print(f'-- {len(self.data_scaled) / init_number_of_rows * 100:.2f}% of the data remains')

    def export_data(self, path_out: str):
        """
        Function to export the data

        :param path_out: str: Path to the output file

        :return: None
        """

        # Define file names
        train_path = os.path.join(path_out, 'train.csv')
        test_path = os.path.join(path_out, 'test.csv')

        self.train.to_csv(train_path, index=True)
        self.test.to_csv(test_path, index=True)

    def compute_correlation(self) -> None:
        """
        Function to compute the correlation between the input and output data

        :return: None
       """

        self.correlation = self.data_scaled.corr()

        na_cols = self.correlation.columns[self.correlation.isna().all()].tolist()
        self.data_scaled.drop(columns=na_cols, inplace=True)
        self.train.drop(columns=na_cols, inplace=True)
        self.test.drop(columns=na_cols, inplace=True)
        self.correlation = self.data_scaled.corr()
        self.p_value = self.data_scaled.corr(method=lambda x, y: linregress(x, y).pvalue)

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

                color: str = 'black' if np.abs(self.correlation.iloc[i, j]) > 0.7 else 'white'
                if self.p_value.iloc[i, j] < 0.01:
                    suffix = '**'
                elif self.p_value.iloc[i, j] < 0.05:
                    suffix = '*'
                else:
                    suffix = ''
                text = ax.text(j, i, str(np.around(self.correlation.iloc[i, j], decimals=2)),
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

        if self.data_scaled is None:
            raise ValueError('Data not loaded')

        if x not in self.data_scaled.columns:
            raise ValueError(f'{x} not in input data')

        x_values = self.data_scaled.loc[:, x].to_numpy().reshape(-1, 1)
        y_values = self.data_scaled.iloc[:, -1].to_numpy().reshape(-1, 1)

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

        if self.data_scaled is None:
            raise ValueError('Data not loaded')

        if column not in self.data_scaled.columns:
            raise ValueError(f'{column} not in input data')

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

        # Plot the histogram
        sns.histplot(self.data_scaled[column], ax=ax1, kde=False, element="bars", edgecolor=None, color='gray',
                     label='Bins')
        ax1.set_xlabel(bef_dict[column], fontsize=20)
        ax1.set_ylabel(f'Frequency', fontsize=20)
        ax1.tick_params(axis='both', colors='black', labelsize=12)

        # Create a secondary y-axis for the KDE plot
        ax2 = ax1.twinx()
        sns.kdeplot(self.data_scaled[column], color='black', linestyle='-.', linewidth=1.5, ax=ax2,
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

        if self.data_scaled is None:
            raise ValueError('Data not loaded')

        if column not in self.data_scaled.columns:
            raise ValueError(f'{column} not in input data')

        fig, ax = plt.subplots(1, 1, figsize=(4, 10))

        b = sns.boxplot(
            y=self.data_scaled[column],
            ax=ax,
            flierprops={"marker": "x"},
            boxprops={"facecolor": "None"},
            linewidth=0.8,
            showfliers=showfliers)

        ax.set_ylabel(bef_dict[column], fontsize=16)
        ax.set_yticks(ax.get_yticks())  # Add this line
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
        sns.histplot(self.data_scaled.iloc[:, -1], ax=ax1, kde=False, element="bars", edgecolor=None, color='gray',
                     label='Bins')
        ax1.set_xlabel(output_label, fontsize=20)
        ax1.set_ylabel('Frequency', fontsize=20)
        ax1.tick_params(axis='both', colors='black', labelsize=12)

        # Create a secondary y-axis for the KDE plot
        ax2 = ax1.twinx()
        sns.kdeplot(self.data_scaled.iloc[:, -1], color='red', linewidth=1.5, linestyle='-.', ax=ax2,
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

        if self.r_2 is None:
            for column in self.data_scaled.iloc[:, :-1].columns:
                self.compute_2d_correlation(column)

        # Get columns where r_2 > 0.3
        columns = [key for key, value in self.r_2.items() if value > threshold]

        print(f'Columns with R^2 > {threshold}: {columns}')

        self.__post_init__()
        self.pre_process_data(**model_params, input_columns=columns)

    def plot_r2(self, save_path: str | None = None, show_plot: bool = False, thresh: float = 0.25) -> None:
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

        ax.set_xticks(range(len(x_values)))  # Add this line
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=20)
        ax.axhline(y=thresh, color='red', linestyle='--', linewidth=1.5, label=f'Threshold: {thresh}')

        ax.tick_params(axis='both', colors='black', labelsize=12)

        # Add the actual R^2 values inside the bars or on top if too small
        for bar, value in zip(bars, y_values):
            height = bar.get_height()
            if height < 0.05:  # Adjust this threshold as needed
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{value:.2f}', ha='center', va='bottom',
                        fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{value:.2f}', ha='center', va='center',
                        fontsize=10, color='white')

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot is True:
            plt.show()

        plt.close()

    def plot_data_exploration(self, data_dir: str, threshold: float = 0.25):
        """
        Function to plot the data exploration

        :return: None
        """

        # Define the path to save the plots
        path_corr_matrix = os.path.join(data_dir, 'correlation_matrices')
        path_2d_corr = os.path.join(data_dir, 'correlations')
        path_distribution = os.path.join(data_dir, 'distributions')
        path_r2 = os.path.join(data_dir, 'r2_values')
        path_boxplots = os.path.join(data_dir, 'boxplots')

        if self.correlation is None:
            self.compute_correlation()

        self.plot_correlation_matrix(save_path=os.path.join(path_corr_matrix, 'correlation_matrix.png'),
                                           show_plot=True)

        for column in self.data_scaled.iloc[:, :-1].columns:
            filename_2d_corr = f'{column}_2d_correlation.png'
            filename_distribution = f'{column}_distribution.png'
            filename_boxplot = f'{column}_boxplot.png'

            self.compute_2d_correlation(column, save_path=os.path.join(path_2d_corr, filename_2d_corr),
                                              show_plot=show_plots)

            self.plot_distribution(column, save_path=os.path.join(path_distribution, filename_distribution),
                                         show_plot=show_plots)

            self.plot_boxplots(column, save_path=os.path.join(path_boxplots, filename_boxplot), show_plot=show_plots,
                                     showfliers=False)

        self.plot_output_distribution(save_path=path_distribution, show_plot=show_plots)
        self.plot_r2(save_path=os.path.join(path_r2, 'r2.png'), show_plot=show_plots, thresh=threshold)


if __name__ == '__main__':

    column_threshold: float = 0.0001
    # Define the model
    data_model = DataModel()

    # ----------------------------------------------
    # --  Explorative Data Analysis (EDA)         --
    # ----------------------------------------------

    pre_processing_params = {'scaler': MinMaxScaler(), 'test_size': 0.2, 'z_value': 6, 'outlier_filter': 'iqr'}
    data_model.pre_process_data(**pre_processing_params, input_columns='all')

    #data_model.plot_data_exploration(data_dir=os.path.join(data_path, 'results', 'data_exploration_full'), threshold=column_threshold)

    # ----------------------------------------------
    # --  Select the relevant columns             --
    # ----------------------------------------------

    data_model.filter_columns(threshold=column_threshold, model_params=pre_processing_params)

    # Repeat the pre-processing step with the selected columns

    data_model.export_data(cnn_data_path)
    data_model.plot_data_exploration(data_dir=os.path.join(data_path, 'results', 'data_exploration_cols'), threshold=column_threshold)
