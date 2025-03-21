'''
Phase 1: Data Exploration and Preprocessing
Data Inventory & Visualization

Temperature Data (Training_data_uhi_index.csv):
Load the 11,229 data points and visualize the UHI index distribution (e.g., histogram and spatial scatter map).
Check for outliers or anomalies in UHI values.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
data_path = r"e:\01-Projects\2025datachallenge\src\Training_data_uhi_index.csv"
df = pd.read_csv(data_path)

# Convert the 'datetime' column to datetime format (adjust format if needed)
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')

# Print basic data summary
print("Data Summary:")
print(df.describe())

# Plot UHI Index histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['UHI Index'], bins=30, kde=True)
plt.title("Distribution of UHI Index")
plt.xlabel("UHI Index")
plt.ylabel("Frequency")
plt.show()

# Scatter plot for spatial distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='UHI Index', data=df, palette='coolwarm', legend=True)
plt.title("Spatial Distribution of UHI Index")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Boxplot for outlier detection
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['UHI Index'])
plt.title("Boxplot of UHI Index")
plt.xlabel("UHI Index")
plt.show()

# Identify outliers using the IQR method
Q1 = df['UHI Index'].quantile(0.25)
Q3 = df['UHI Index'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['UHI Index'] < lower_bound) | (df['UHI Index'] > upper_bound)]

print("Detected outliers:")
print(outliers)
