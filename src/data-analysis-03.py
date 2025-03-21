import pandas as pd
import matplotlib.pyplot as plt

# Path to the Excel file
excel_file = r"e:\01-Projects\2025datachallenge\src\NY_Mesonet_Weather.xlsx"

# Load the Excel file and list available sheets
xlsx = pd.ExcelFile(excel_file)
print("Available sheets:", xlsx.sheet_names)

# Load summary sheet if it exists
if 'Summary' in xlsx.sheet_names:
    df_summary = pd.read_excel(excel_file, sheet_name='Summary')
    print("Weather Summary:")
    print(df_summary.head())
else:
    print("No 'Summary' sheet found.")

# Function to load and preprocess a weather sheet (Manhattan or Bronx)
def load_weather_sheet(sheet_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    # Convert the 'Date / Time' column to datetime (removing timezone text if needed)
    df['Date / Time'] = pd.to_datetime(df['Date / Time'].str.replace(' EDT', ''), errors='coerce')
    return df

# Load Manhattan and Bronx sheets
df_manhattan = load_weather_sheet('Manhattan')
df_bronx = load_weather_sheet('Bronx')

print("Manhattan Weather Data:")
print(df_manhattan.head())

print("Bronx Weather Data:")
print(df_bronx.head())

# Plot time series for Manhattan weather data
plt.figure(figsize=(12, 6))
plt.plot(df_manhattan['Date / Time'], df_manhattan['Air Temp at Surface [degC]'], label='Air Temp [degC]')
plt.plot(df_manhattan['Date / Time'], df_manhattan['Avg Wind Speed [m/s]'], label='Avg Wind Speed [m/s]')
plt.plot(df_manhattan['Date / Time'], df_manhattan['Solar Flux [W/m^2]'], label='Solar Flux [W/m^2]')
plt.xlabel("Date / Time")
plt.ylabel("Values")
plt.title("Manhattan Weather Time Series")
plt.legend()
plt.tight_layout()
plt.show()

# Plot time series for Bronx weather data
plt.figure(figsize=(12, 6))
plt.plot(df_bronx['Date / Time'], df_bronx['Air Temp at Surface [degC]'], label='Air Temp [degC]')
plt.plot(df_bronx['Date / Time'], df_bronx['Avg Wind Speed [m/s]'], label='Avg Wind Speed [m/s]')
plt.plot(df_bronx['Date / Time'], df_bronx['Solar Flux [W/m^2]'], label='Solar Flux [W/m^2]')
plt.xlabel("Date / Time")
plt.ylabel("Values")
plt.title("Bronx Weather Time Series")
plt.legend()
plt.tight_layout()
plt.show()