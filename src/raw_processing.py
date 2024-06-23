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
"""

data_path = '../data/pre_processed/interpolated_ds.csv'
outfile = '../data/pre_processed/full_ds.csv'
data = pd.read_csv(data_path)

# Calculate the new columns
if 'lotarea' in data.columns and 'bldgarea' in data.columns:
    data['lotbldgratio'] = data['bldgarea'] / data['lotarea']

if 'bldgarea' in data.columns and 'numfloors' in data.columns:
    data['floorarearatio'] = data['bldgarea'] / data['numfloors']

if 'unitstotal' in data.columns and 'bldgarea' in data.columns:
    data['unitarea'] = data['bldgarea'] / data['unitstotal']

if 'resarea' in data.columns and 'bldgarea' in data.columns:
    data['resratio'] = data['resarea'] / data['bldgarea']

if 'comarea' in data.columns and 'bldgarea' in data.columns:
    data['comratio'] = data['comarea'] / data['bldgarea']

if 'sarea' in data.columns and 'volume' in data.columns:
    data['sareavolumeratio'] = data['sarea'] / data['volume']

if 'sarea' in data.columns and 'bldgarea' in data.columns:
    data['sareabldgratio'] = data['sarea'] / data['bldgarea']

if 'sarea' in data.columns and 'lotarea' in data.columns:
    data['sarealotratio'] = data['sarea'] / data['lotarea']

if 'volume' in data.columns and 'bldgarea' in data.columns:
    data['volumebldgratio'] = data['volume'] / data['bldgarea']

if 'volume' in data.columns and 'lotarea' in data.columns:
    data['volumelotratio'] = data['volume'] / data['lotarea']




# Insert the new columns at specific positions
data.insert(10, 'lotbldgratio', data.pop('lotbldgratio'))
data.insert(11, 'floorarearatio', data.pop('floorarearatio'))
data.insert(12, 'unitarea', data.pop('unitarea'))
data.insert(13, 'resratio', data.pop('resratio'))
data.insert(14, 'comratio', data.pop('comratio'))
data.insert(15, 'sareavolumeratio', data.pop('sareavolumeratio'))
data.insert(16, 'sareabldgratio', data.pop('sareabldgratio'))
data.insert(17, 'sarealotratio', data.pop('sarealotratio'))
data.insert(18, 'volumebldgratio', data.pop('volumebldgratio'))
data.insert(19, 'volumelotratio', data.pop('volumelotratio'))

#data.drop(columns=['irrlotcode'], inplace=True)



print(data.columns)


data.to_csv(outfile, index=False)


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

