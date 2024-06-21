import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import scienceplots

plt.style.use(['science', 'ieee'])

# Load the dataset
file_path = '../data/pre_processed/in_and_out_filtered.csv'
out_path = '../data/pre_processed/in_and_out_filled_vol.csv'
plot_path = '../data/results/interpolation/interpolation.png'
data = pd.read_csv(file_path)

# Separate the data with and without volume values
data_with_volume = data[data['Volume'] > 0]
data_without_volume = data[data['Volume'] == 0]

# Extract the relevant columns
X = data_with_volume[['SArea']].values.reshape(len(data_with_volume[['SArea']].values), )
y = data_with_volume['Volume'].values

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(len(X), )
y = scaler.fit_transform(y.reshape(-1, 1)).reshape(len(y), )

print(X.shape, y.shape)

# Fit polynomial regression model
p = np.polyfit(X, y, 2)
y_p = np.polyval(p, np.linspace(np.min(X), np.max(X), 1000))

# RMSE
rmse = np.sqrt(np.mean((y - np.polyval(p, X)) ** 2))
print(f'RMSE: {rmse}')

# R^2
r2 = 1 - (np.sum((y - np.polyval(p, X)) ** 2) / np.sum((y - np.mean(y)) ** 2))
print(f'R^2: {r2}')


# Create a new X range for predictions
X_new = np.linspace(np.min(X), np.max(X), 1000)
y_fit = np.polyval(p, X_new)


fig, ax = plt.subplots(1, 1, figsize=(10, 9))
ax.tick_params(axis='both', colors='black', labelsize=12)
ax.scatter(X, y, marker='x', s=15, alpha=0.5, label='Data Points')
ax.plot(X_new, y_fit, color='red', label='Polynomial Fit', linestyle='--')
ax.set_xlabel('Surface Area', fontsize=16)
ax.set_ylabel('Volume', fontsize=16)
ax.set_title(r'${}^{}={}$'.format('R', 2, np.round(r2, 2)), fontsize=24)
plt.legend(fontsize=16)
plt.savefig(plot_path)


data[data['Volume'] == 0] = data[data['Volume'] == 0].assign(Volume=np.polyval(p, data[data['Volume'] == 0]['SArea']))

# Save the data
data.to_csv(out_path, index=False)
