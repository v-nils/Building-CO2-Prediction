import pandas as pd
import matplotlib.pyplot as plt


def change_decimal_sign(value):
    if pd.isna(value):  # Check for NaN values
        return value
    return value.replace('.', ',') if '.' in value else value.replace(',', '.')


columns_to_convert = ['SArea', 'Volume']

converters = {col: change_decimal_sign for col in columns_to_convert}

df = pd.read_csv('../data/pre_processed/building_vol_sarea_sample.csv', delimiter=';', converters=converters)

for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

print(df.head())

# Normalize the data
df['SArea'] = df['SArea'] / df['SArea'].max()
df['Volume'] = df['Volume'] / df['Volume'].max()

fig, ax = plt.subplots(1,1, figsize=(10, 10))
ax.scatter(df['SArea'], df['Volume'])
plt.show()