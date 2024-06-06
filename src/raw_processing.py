import geopandas as gpd
import pandas.io.sql as psql
import os
from db_connect import create_db_engine


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

