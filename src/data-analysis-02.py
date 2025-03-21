'''
Phase 1: Data Exploration and Preprocessing
Data Inventory & Visualization

Building Footprints (Building_Footprint.kml):
Load the KML file and visualize building distributions.
Determine potential metrics (e.g., building density or impervious area) by creating buffers around the UHI points.
'''
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

# Load the CSV data
data_path = r"e:\01-Projects\2025datachallenge\src\Training_data_uhi_index.csv"
df = pd.read_csv(data_path)

# Load the building footprints KML file
kml_path = r"e:\01-Projects\2025datachallenge\src\Building_Footprint.kml"
buildings = gpd.read_file(kml_path, driver='KML')

print("Building Footprints Summary:")
print(buildings.head())

# Visualize building footprints
fig, ax = plt.subplots(figsize=(10, 6))
buildings.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)
plt.title("Building Footprints")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# --- Creating buffers around UHI points ---
# Convert the UHI DataFrame to a GeoDataFrame using proper column names for coordinates
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
uhi_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Reproject to a projected CRS (e.g., UTM zone 18N for NYC)
uhi_gdf_proj = uhi_gdf.to_crs(epsg=32618)

# Create a buffer in meters (e.g., 500 m)
uhi_gdf_proj['buffer'] = uhi_gdf_proj.geometry.buffer(500)

# Convert the buffered geometries back to original geographic CRS for visualization/analysis
uhi_gdf['buffer'] = uhi_gdf_proj['buffer'].to_crs(uhi_gdf.crs)

# For visualization, plot the UHI points, buffers, and building footprints on one map
fig, ax = plt.subplots(figsize=(10, 6))
buildings.to_crs(uhi_gdf.crs).plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)
uhi_gdf.plot(ax=ax, color='red', markersize=2)
uhi_gdf['buffer'].plot(ax=ax, color='none', edgecolor='green', linewidth=0.5, alpha=0.5)
plt.title("UHI Points with Buffers and Building Footprints")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# --- Determine potential metrics: building density ---
# Reproject building footprints to match UHI GeoDataFrame CRS
buildings = buildings.to_crs(uhi_gdf.crs)

# Define a function that counts building footprints intersecting a given buffer
def count_buildings_in_buffer(buffer_geom):
    return buildings[buildings.intersects(buffer_geom)].shape[0]

uhi_gdf['building_count'] = uhi_gdf['buffer'].apply(count_buildings_in_buffer)

print("UHI points with building counts:")
print(uhi_gdf[['Longitude', 'Latitude', 'UHI Index', 'building_count']].head())