import geopandas as gpd
import numpy as np
import pandas.io.sql as psql
import pandas as pd
import os
from db_connect import create_db_engine

"""
# GLOBAL VARIABLES
project_path: str = os.path.abspath(__file__)
root_path: str = os.path.dirname(os.path.dirname(project_path))

data_path: str = os.path.join(root_path, r'data\raw')

pluto_path: str = os.path.join(data_path, r'MapPLUTO24v1_1.gdb')
energy_path: str = os.path.join(data_path, r'energy_data.csv')


engine = create_db_engine()

sql_input = str('SELECT * FROM bef_input')
sql_output = str('SELECT * FROM energy_output')

with engine.connect() as connection:
    input_ = gpd.GeoDataFrame.from_postgis(sql_input, connection, geom_col='geom', index_col='bef_id')
    output_ = psql.read_sql(sql_output, connection, index_col='energy_id')

save_path_bef: str = os.path.join(root_path, r'data\pre_processed\bef_input.csv')
save_path_energy: str = os.path.join(root_path, r'data\pre_processed\energy_output.csv')

input_.to_csv(save_path_bef)
output_.to_csv(save_path_energy)


data_path = '../data/pre_processed/in_and_out_filled_vol.csv'
outfile = '../data/pre_processed/bef_energy_data.csv'
data = pd.read_csv(data_path)

if 'lot_area' in data.columns and 'building_area' in data.columns:
    data.loc[:, 'lot_bldg_ratio'] = data['building_area'] / data['lot_area']

# Floor area ratio
if 'building_area' in data.columns and 'num_floors' in data.columns:
    data.loc[:, 'floor_area_ratio'] = data['building_area'] / data[
        'num_floors']

# Area per unit
if 'total_units' in data.columns and 'building_area' in data.columns:
    data.loc[:, 'unit_area'] = data['building_area'] / data['total_units']

# Ratio residential to commercial area
if 'residential_area' in data.columns and 'building_area' in data.columns:
    data.loc[:, 'res_ratio'] = data['residential_area'] / data[
        'building_area']

# Ration commercial area to total area
if 'commercial_area' in data.columns and 'building_area' in data.columns:
    data.loc[:, 'com_ratio'] = data['commercial_area'] / data[
        'building_area']

# Ratio of residential area to total area
if 'SArea' in data.columns and 'Volume' in data.columns:
    data.loc[:, 'sarea_volume_ratio'] = data['SArea'] / data['Volume']


data.to_csv(outfile, index=False)"""


def convert_columns_to_valid_dbnames(list_of_columns):

    new_list = []

    for word in list_of_columns:

        # change all to lower case
        word = word.lower()

        # check if the word contains any special characters
        if not word.isalnum():


            # remove all special characters
            word = ''.join(e for e in word if e.isalnum() or e == '_')

            # check if the first character is a digit
            if word[0].isdigit():

                # add an underscore at the beginning
                word = '_' + word

        word = word[:63]

        print("Word: ", word)
        new_list.append(word)

    # Check for duplicates
    for word in new_list:

        if new_list.count(word) > 1:
            print("Duplicate: ", word)


    return new_list


df = pd.read_csv(r'C:\Users\nilse\Documents\projects\LBS\data\raw\energy_data_filtered.csv', encoding='latin1', index_col=False)

old_cols = df.columns

cols = convert_columns_to_valid_dbnames(list(old_cols))
col_dict = dict(zip(old_cols, cols))
print(col_dict)
df.rename(columns=col_dict, inplace=True)
print(df.head())
df.loc[df['property_gfa___self_reported__ft__'].isin(['No', 'Yes']), 'property_gfa___self_reported__ft__'] = np.nan
df[df['parent_property_id'] == '4926122, 15143466'] = np.nan
df[df['natural_gas_use__kbtu_'] == 'Insufficient access'] = np.nan

df.to_csv(r'C:\Users\nilse\Documents\projects\LBS\data\raw\energy_data_filtered_utf8.csv', columns=cols, index=False, encoding='UTF-8')
