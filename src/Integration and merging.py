# --- Integration: Align and Merge Datasets ---

import geopandas as gpd
import pandas as pd
import math
# from shapely.geometry import Point
# from datetime import datetime

# 1. Load and prepare the ground-truth UHI data
uhi_df = pd.read_csv("Training_data_uhi_index.csv")
# Convert datetime column (adjust format if needed)
uhi_df["datetime"] = pd.to_datetime(uhi_df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
# Create a GeoDataFrame from UHI data
uhi_gdf = gpd.GeoDataFrame(uhi_df, 
                           geometry=gpd.points_from_xy(uhi_df.Longitude, uhi_df.Latitude),
                           crs="EPSG:4326")
# (If needed, reproject to match your building grid CRS; here we use EPSG:2263 as an example)
uhi_gdf = uhi_gdf.to_crs(epsg=2263)

# 2. Load the building density/grid data
# (Assuming you exported the 100 m grid from examining_building_footprints.py as a shapefile)
grid = gpd.read_file("grid_200m.shp")
# Ensure the grid has the same CRS as UHI points
grid = grid.to_crs(uhi_gdf.crs)

# Perform a spatial join to assign building features to each UHI measurement
# (This is a point-to-vector join: for each UHI point, find the grid cell it falls within)
uhi_with_buildings = gpd.sjoin(uhi_gdf, grid[['building_d', 'impervious', 'geometry']],
                               how="left", predicate="within")

# 3. Load and prepare the weather datasets

## Manhattan weather data
weather_manhattan = pd.read_excel("NY_Mesonet_Weather.xlsx", sheet_name="Manhattan")
# Convert the 'Date / Time' column to datetime (remove ' EDT' if present)
weather_manhattan['Date / Time'] = pd.to_datetime(weather_manhattan['Date / Time'].str.replace(' EDT', ''))
# Filter to only include measurements from 3:00pm to 4:00pm
weather_manhattan = weather_manhattan[(weather_manhattan['Date / Time'].dt.hour >= 15) & (weather_manhattan['Date / Time'].dt.hour < 16)]
                
## Bronx weather data
weather_bronx = pd.read_excel("NY_Mesonet_Weather.xlsx", sheet_name="Bronx")
weather_bronx['Date / Time'] = pd.to_datetime(weather_bronx['Date / Time'].str.replace(' EDT', ''))
# Filter to only include measurements from 3:00pm to 4:00pm
weather_bronx = weather_bronx[(weather_bronx['Date / Time'].dt.hour >= 15) & (weather_bronx['Date / Time'].dt.hour < 16)]

# Helper function to compute the haversine distance (in kilometers)
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Define the coordinates for the two weather stations:
manhattan_coord = (-73.96449, 40.76754)  # (Longitude, Latitude)
bronx_coord    = (-73.89352, 40.87248)    # (Longitude, Latitude)

# Helper function to get the closest weather observation for a given datetime (dt)
def get_closest_weather(weather_df, dt):
    # Compute absolute time difference between each observation and dt
    time_diffs = (weather_df['Date / Time'] - dt).abs()
    # Get the row index with the smallest difference
    idx_closest = time_diffs.idxmin()
    return weather_df.loc[idx_closest, ['Air Temp at Surface [degC]', 'Avg Wind Speed [m/s]', 'Solar Flux [W/m^2]']]

# Function to assign weather values based on location and nearest time observation.
def assign_weather(row):
    dt = row['datetime']  # UHI measurement time (captured at 1-min intervals)
    d_man = haversine(row['Longitude'], row['Latitude'], manhattan_coord[0], manhattan_coord[1])
    d_bron = haversine(row['Longitude'], row['Latitude'], bronx_coord[0], bronx_coord[1])
    if d_man <= d_bron:
        weather_vals = get_closest_weather(weather_manhattan, dt)
    else:
        weather_vals = get_closest_weather(weather_bronx, dt)
    return pd.Series({
       'avg_temp': weather_vals['Air Temp at Surface [degC]'],
       'avg_wind_speed': weather_vals['Avg Wind Speed [m/s]'],
       'avg_solar_flux': weather_vals['Solar Flux [W/m^2]']
    })

# 4. Attach dynamic weather information to each UHI point in the integrated dataset
# Instead of global averages, we now update each row based on its timestamp.
weather_assigned = uhi_with_buildings.apply(assign_weather, axis=1)
uhi_with_buildings = pd.concat([uhi_with_buildings, weather_assigned], axis=1)

# (Optional) Drop the original separate weather columns if no longer needed.
# uhi_with_buildings = uhi_with_buildings.drop(columns=[
#    'avg_temp_manhattan', 'avg_wind_speed_manhattan', 'avg_solar_flux_manhattan',
#    'avg_temp_bronx', 'avg_wind_speed_bronx', 'avg_solar_flux_bronx'
#])

# 5. (Optional) Integrate Satellite-derived indices
# If you have a function like map_satellite_data() (see UHI_Experiment_Sample_Benchmark_Notebook_V5.py),
# you can extract satellite features at UHI point locations. For example:
# sat_df = map_satellite_data("S2_sample_01.tiff", "Training_data_uhi_index.csv")
# Then, merge using the UHI DataFrame's index or via spatial join if coordinates are provided.
# For illustration, let’s assume sat_df has the same order as uhi_df and contains an 'NDVI' column:
# uhi_with_buildings["NDVI_satellite"] = sat_df["NDVI"].values


print("Integrated Dataset Head with unified, time-specific weather data:")
print(uhi_with_buildings.head())

# 5. Save or inspect the integrated dataset
uhi_with_buildings.to_csv("Integrated_UHI_Dataset_200m.csv", index=False)


# or, to preserve spatial geometries: uhi_with_buildings.to_file("Integrated_UHI_Dataset.shp")

