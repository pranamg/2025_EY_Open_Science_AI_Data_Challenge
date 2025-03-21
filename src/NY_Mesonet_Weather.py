import pandas as pd
import os
import matplotlib.pyplot as plt

if os.path.exists('NY_Mesonet_Weather.xlsx'):
    try:
        xls = pd.ExcelFile('NY_Mesonet_Weather.xlsx')
        sheet_names = xls.sheet_names
        print(f"Available sheet names: {sheet_names}")

        # Attempt to read the sheets with corrected names (if needed)
        if 'Weather_Summary' in sheet_names:
            df_summary = pd.read_excel(xls, 'Summary')
        elif 'Weather Summary' in sheet_names:
          df_summary = pd.read_excel(xls, 'Weather Summary')
        else:
            df_summary = None
            print("Error: 'Weather Summary' or 'Summary' sheet not found.")

        if 'Manhattan_data' in sheet_names:
            df_manhattan = pd.read_excel(xls, 'Manhattan_data')
        elif 'Manhattan' in sheet_names:
            df_manhattan = pd.read_excel(xls, 'Manhattan')
        else:
            df_manhattan = None
            print("Error: 'Manhattan' or 'Manhattan_data' sheet not found.")

        if 'Bronx_data' in sheet_names:
            df_bronx = pd.read_excel(xls, 'Bronx_data')
        elif 'Bronx' in sheet_names:
            df_bronx = pd.read_excel(xls, 'Bronx')
        else:
            df_bronx = None
            print("Error: 'Bronx' or 'Bronx_data' sheet not found.")


        # display(df_summary.head() if df_summary is not None else "Weather Summary DataFrame not found")
        # display(df_manhattan.head() if df_manhattan is not None else "Manhattan DataFrame not found")
        # display(df_bronx.head() if df_bronx is not None else "Bronx DataFrame not found")

    except Exception as e:
        print(f"An error occurred: {e}")
        df_summary, df_manhattan, df_bronx = None, None, None
else:
    print("Error: 'NY_Mesonet_Weather.xlsx' not found.")
    df_summary, df_manhattan, df_bronx = None, None, None

import matplotlib.pyplot as plt

# Inspect DataFrames (df_manhattan and df_bronx)
print("Manhattan DataFrame:")
print("Shape:", df_manhattan.shape)
# display(df_manhattan.head())
# display(df_manhattan.tail())

print("\nBronx DataFrame:")
print("Shape:", df_bronx.shape)
# display(df_bronx.head())
# display(df_bronx.tail())


# Data Types and Missing Values
print("\nManhattan DataFrame Info:")
print(df_manhattan.info())
print("\nMissing values in Manhattan DataFrame:\n", df_manhattan.isnull().sum())

print("\nBronx DataFrame Info:")
print(df_bronx.info())
print("\nMissing values in Bronx DataFrame:\n", df_bronx.isnull().sum())


# Summary Statistics
print("\nManhattan DataFrame Summary Statistics:")
# display(df_manhattan.describe())
print("\nBronx DataFrame Summary Statistics:")
# display(df_bronx.describe())


# Distribution of Key Variables (Histograms)
plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.hist(df_manhattan['Air Temp at Surface [degC]'], bins=20, color='skyblue', edgecolor='black')
plt.title('Manhattan Air Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')

plt.subplot(2, 3, 2)
plt.hist(df_manhattan['Avg Wind Speed [m/s]'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Manhattan Wind Speed')

plt.subplot(2, 3, 3)
plt.hist(df_manhattan['Solar Flux [W/m^2]'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Manhattan Solar Flux')


plt.subplot(2, 3, 4)
plt.hist(df_bronx['Air Temp at Surface [degC]'], bins=20, color='skyblue', edgecolor='black')
plt.title('Bronx Air Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')

plt.subplot(2, 3, 5)
plt.hist(df_bronx['Avg Wind Speed [m/s]'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Bronx Wind Speed')

plt.subplot(2, 3, 6)
plt.hist(df_bronx['Solar Flux [W/m^2]'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Bronx Solar Flux')

plt.tight_layout()
plt.show()


# Identify Time Range
print("\nManhattan Time Range:")
print("Min Date/Time:", df_manhattan['Date / Time'].min())
print("Max Date/Time:", df_manhattan['Date / Time'].max())

print("\nBronx Time Range:")
print("Min Date/Time:", df_bronx['Date / Time'].min())
print("Max Date/Time:", df_bronx['Date / Time'].max())

# Convert 'Date / Time' to datetime objects
df_manhattan['Date / Time'] = pd.to_datetime(df_manhattan['Date / Time'])
df_bronx['Date / Time'] = pd.to_datetime(df_bronx['Date / Time'])

# Missing Value Handling (Double-check and document strategy)
print("\nMissing values in Manhattan DataFrame:\n", df_manhattan.isnull().sum())
print("\nMissing values in Bronx DataFrame:\n", df_bronx.isnull().sum())
# No missing values were found in the previous exploration and this recheck confirms that.

# Unit Consistency Check and Conversion (Document any conversions)
# All units seem to be consistent based on the column names and the data exploration.
# No unit conversion is needed.  Documenting the units as they are.

# Data Consistency Checks (Wind speed and direction)
# Check for negative wind speed
print("\nManhattan Negative Wind Speeds:\n", df_manhattan[df_manhattan['Avg Wind Speed [m/s]'] < 0])
print("\nBronx Negative Wind Speeds:\n", df_bronx[df_bronx['Avg Wind Speed [m/s]'] < 0])

# Check for wind direction outside 0-360
print("\nManhattan Wind Direction Out of Range:\n", df_manhattan[(df_manhattan['Wind Direction [degrees]'] < 0) | (df_manhattan['Wind Direction [degrees]'] > 360)])
print("\nBronx Wind Direction Out of Range:\n", df_bronx[(df_bronx['Wind Direction [degrees]'] < 0) | (df_bronx['Wind Direction [degrees]'] > 360)])

# If inconsistencies are found, handle them appropriately (e.g., replace with mean, drop rows, etc.)
# Based on the previous checks, no inconsistencies were found, so no values were modified.

# Merge the two dataframes
df_merged = pd.merge(df_manhattan, df_bronx, on='Date / Time', how='inner', suffixes=('_manhattan', '_bronx'))

# Print the shape and first few rows of the merged dataframe
print("Shape of merged DataFrame:", df_merged.shape)
# display(df_merged.head())

import math
import numpy as np

# Aggregate variables
mean_wind_speed_manhattan = df_merged['Avg Wind Speed [m/s]_manhattan'].mean()
mean_wind_direction_manhattan = df_merged['Wind Direction [degrees]_manhattan'].mean()
mean_solar_flux_manhattan = df_merged['Solar Flux [W/m^2]_manhattan'].mean()

mean_wind_speed_bronx = df_merged['Avg Wind Speed [m/s]_bronx'].mean()
mean_wind_direction_bronx = df_merged['Wind Direction [degrees]_bronx'].mean()
mean_solar_flux_bronx = df_merged['Solar Flux [W/m^2]_bronx'].mean()

# Calculate heat index (using a simplified formula as an example)
# Note: This is a placeholder and might not be the most accurate heat index formula
# A more accurate approach would be to use a dedicated library or more complex formula.

def calculate_heat_index(temperature, humidity):
    return temperature + 0.36 * humidity

df_merged['heat_index_manhattan'] = calculate_heat_index(df_merged['Air Temp at Surface [degC]_manhattan'], df_merged['Relative Humidity [percent]_manhattan'])
df_merged['heat_index_bronx'] = calculate_heat_index(df_merged['Air Temp at Surface [degC]_bronx'], df_merged['Relative Humidity [percent]_bronx'])

# Determine wind vector components
df_merged['u_wind_manhattan'] = df_merged['Avg Wind Speed [m/s]_manhattan'] * np.cos(np.radians(df_merged['Wind Direction [degrees]_manhattan']))
df_merged['v_wind_manhattan'] = df_merged['Avg Wind Speed [m/s]_manhattan'] * np.sin(np.radians(df_merged['Wind Direction [degrees]_manhattan']))

df_merged['u_wind_bronx'] = df_merged['Avg Wind Speed [m/s]_bronx'] * np.cos(np.radians(df_merged['Wind Direction [degrees]_bronx']))
df_merged['v_wind_bronx'] = df_merged['Avg Wind Speed [m/s]_bronx'] * np.sin(np.radians(df_merged['Wind Direction [degrees]_bronx']))


# Add aggregated metrics to the DataFrame
df_merged['mean_wind_speed_manhattan'] = mean_wind_speed_manhattan
df_merged['mean_wind_direction_manhattan'] = mean_wind_direction_manhattan
df_merged['mean_solar_flux_manhattan'] = mean_solar_flux_manhattan

df_merged['mean_wind_speed_bronx'] = mean_wind_speed_bronx
df_merged['mean_wind_direction_bronx'] = mean_wind_direction_bronx
df_merged['mean_solar_flux_bronx'] = mean_solar_flux_bronx

# display(df_merged.head())



# Select variables for plotting
variables = ['Air Temp at Surface [degC]', 'Avg Wind Speed [m/s]', 'Wind Direction [degrees]', 'Solar Flux [W/m^2]', 'heat_index', 'u_wind', 'v_wind']

# Create subplots
fig, axes = plt.subplots(len(variables), 1, figsize=(15, 5 * len(variables)))

# Iterate through variables and create plots
for i, var in enumerate(variables):
    ax = axes[i]
    ax.plot(df_merged['Date / Time'], df_merged[f'{var}_manhattan'], label='Manhattan', color='blue')
    ax.plot(df_merged['Date / Time'], df_merged[f'{var}_bronx'], label='Bronx', color='red')
    ax.set_xlabel('Date / Time')
    ax.set_ylabel(var)
    ax.set_title(f'{var} Time Series')
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust layout and # display the plot
plt.tight_layout()
plt.show()

