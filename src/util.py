import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.preprocessing import StandardScaler


def match_df(df_1: pd.DataFrame, df_2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows from df_1 that are not present in df_2 and vice versa.

    :param df_1: (pd.DataFrame) DataFrame to match
    :param df_2: (pd.DataFrame) DataFrame to match
    :return: (tuple) DataFrames with rows removed
    """

    df_1 = df_1[df_1.index.isin(df_2.index)]
    df_2 = df_2[df_2.index.isin(df_1.index)]

    return df_1, df_2


def remove_outliers_zscore(df: pd.DataFrame, z: float, axis: int = 1) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame using the z-score method.

    :param df: (pd.DataFrame) DataFrame to remove outliers from
    :param z: (float) Z-score threshold
    :param axis: (int) Axis to remove outliers from
    :return: (pd.DataFrame) DataFrame with outliers removed
    """
    print(df)
    offset: float = 1e-10
    df = df.replace([0], offset)
    z_scores = np.abs(stats.zscore(df, axis=axis))
    print(z_scores)
    return df[(z_scores < z).all(axis=1)]


def remove_outliers_iqr(df: pd.DataFrame, q1: float = 0.25, q2: float = 0.75, axis: int = 1) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame using the IQR method.

    :param df: (pd.DataFrame) DataFrame to remove outliers from
    :param q1: (float) Lower quantile
    :param q2: (float) Upper quantile
    :return: (pd.DataFrame) DataFrame with outliers removed
    """

    Q1 = df.quantile(q1)
    Q3 = df.quantile(q2)
    IQR = Q3 - Q1
    print(f'Q1:\n{Q1}')
    print(f'Q3:\n{Q3}')
    print(f'IQR:\n{IQR}')

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f'Lower Bound:\n{lower_bound}')
    print(f'Upper Bound:\n{upper_bound}')

    filtered_df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=axis)]
    return filtered_df


def fit_transform_df(df: pd.DataFrame, scaler: object = StandardScaler) -> pd.DataFrame:
    """
    Fit and transform a DataFrame using a scaler object.

    :param df: (pd.DataFrame) DataFrame to fit and transform
    :param scaler: (object) Scaler object
    :return: (pd.DataFrame) Scaled DataFrame
    """

    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def transform_df(df: pd.DataFrame, scaler: object = StandardScaler) -> pd.DataFrame:
    """
    Transform a DataFrame using a scaler object.

    :param df: (pd.DataFrame) DataFrame to transform
    :param scaler: (object) Scaler object
    :return: (pd.DataFrame) Scaled DataFrame
    """

    return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
