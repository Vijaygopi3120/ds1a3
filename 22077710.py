# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score

df = pd.read_csv('API_AG.LND.AGRI.K2_DS2.csv',skiprows=4)
df.head()

# Check for null values in the dataset
null_values = df.isnull().sum()

# Display columns with null values
columns_with_null = null_values[null_values > 0]
print("Columns with null values:")
print(columns_with_null)

# Remove rows with null values
df_cleaned = df.fillna(0)

df.head()

# Choose the most recent year (e.g., 2021)
selected_year = '2021'
# Select relevant columns for clustering
columns_for_clustering = ['Country Name', 'Indicator Name', selected_year]
# Extract relevant data for clustering
df_selected = df[columns_for_clustering]
# Remove rows with null values
df_selected = df_selected.dropna()
# Normalize the data using StandardScaler
scaler = StandardScaler()
df_normalized = df_selected.copy()
df_normalized[selected_year] = scaler.fit_transform(df_normalized[[selected_year]])
# Specify the number of clusters
num_clusters = 3

# Fit K-means to the normalized data
n_init_value = 10  # Set the desired value for n_init
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=n_init_value)
df_normalized['Cluster'] = kmeans.fit_predict(df_normalized[[selected_year]])

# Add cluster information to the original dataframe
df_clustered = pd.merge(df, df_normalized[['Country Name', 'Cluster']], on='Country Name', how='left')

# Visualization - Scatter plot with Cluster Centers
plt.figure(figsize=(12, 6))
for cluster_label in range(num_clusters):
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_label]
    plt.scatter(cluster_data['Country Name'], cluster_data[selected_year], label=f'Cluster {cluster_label + 1}', s=50)

plt.title(f'Clustering Results for {selected_year}')
plt.xlabel('Country Name')
plt.ylabel(f'Indicator Value for {selected_year}')
plt.legend()
plt.show()

# Extract the features for evaluation
X_eval = df_normalized[[selected_year, 'Cluster']]

# Remove rows with null values in the evaluation set
X_eval = X_eval.dropna()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_eval[[selected_year]], X_eval['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Fit K-means to the normalized data
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=n_init_value)
df_normalized['Cluster'] = kmeans.fit_predict(df_normalized[[selected_year]])

# Calculate inertia
inertia = kmeans.inertia_
print(f"Inertia: {inertia}")



import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
path = 'API_AG.LND.TOTL.UR.K2_DS2.csv'
df = pd.read_csv(path, skiprows=4)
df.head()

# Check for null values in the dataset
null_values = df.isnull().sum()

# Display columns with null values
columns_with_null = null_values[null_values > 0]
print("Columns with null values:")
print(columns_with_null)

# Remove rows with null values
df_cleaned = df.fillna(0)
df_cleaned.head()

df_cleaned.info()

import matplotlib.pyplot as plt

# Selecting data for a particular country
country_data = df_cleaned[df_cleaned['Country Name'] == 'United States']
years = country_data.columns[4:-1].astype(int)
attribute_values = country_data.iloc[:, 4:-1].values.flatten()

# Defining a second-degree polynomial function
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

# Fitting the data utilising curve_fit
params, covariance = curve_fit(polynomial_function, years, attribute_values)

# Generating predicted values for the years 1960-2030
years_future = np.arange(1960, 2031)
predicted_values = polynomial_function(years_future, *params)

# Visualising the original data and the fitted curve
plt.scatter(years, attribute_values, label='Original Data')
plt.plot(years_future, predicted_values, label='Fitted Curve', color='green')
plt.xlabel('Year')
plt.ylabel('Value of the Attribute')
plt.title('Fitting a Second-Degree Polynomial')
plt.legend()
plt.show()
# Predicting values for the year 2030
prediction30 = polynomial_function(2030, *params)
print(f'Predicted value for 2030: {prediction30:.2f}')

# Defining the err_ranges function
def err_ranges(covariance, alpha=0.05):
    alpha = 1 - alpha
    n = len(covariance)
    quantile = 1 - alpha / 2
    err = np.zeros(n)
    for i in range(n):
        err[i] = np.sqrt(covariance[i, i]) * np.abs(np.percentile(np.random.normal(0, 1, 10000), quantile))
    return err


# Fitting the data using curve_fit
params, covariance = curve_fit(polynomial_function, years, attribute_values)


# Estimating confidence range using err_ranges
confidence_range = err_ranges(covariance)


# Repeating confidence_range for each predicted value
confidence_range_broadcasted = np.tile(confidence_range, (len(years_future), 1))


# Visualising the original data
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.scatter(years, attribute_values, label='Original Data')
plt.xlabel('Year')
plt.ylabel('Value of the Attribute')
plt.title('Original Data')

# Select a specific column from confidence_range_broadcasted
confidence_range_column = confidence_range_broadcasted[:, 0]

# Visualising the fitted curve and confidence range
plt.subplot(2, 1, 2)
plt.plot(years_future, predicted_values, label='Fitted Curve', color='green')
plt.fill_between(years_future, predicted_values - confidence_range_column, predicted_values + confidence_range_column, color='gray', alpha=0.2, label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('Value of the Attribute')
plt.title('Fitting a Second-Degree Polynomial using Confidence Range')
plt.legend()

plt.tight_layout()
plt.show()