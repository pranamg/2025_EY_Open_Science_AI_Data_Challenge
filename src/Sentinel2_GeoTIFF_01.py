# Supress Warnings 
import warnings
warnings.filterwarnings('ignore')

# Import common GIS tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
from matplotlib.cm import RdYlGn, jet, RdBu

# Import Planetary Computer tools
import stackstac
import pystac_client
import planetary_computer 
from odc.stac import stac_load

# --- New Imports for Feature Engineering ---
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Define the bounding box for the archive search (for scene retrieval)
# (This can be kept as your original target area for querying Sentinel-2)
lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
time_window = "2021-06-01/2021-09-01"

stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = stac.search(
    bbox=bounds, 
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)
items = list(search.get_items())
print('This is the number of scenes that touch our region:', len(items))
signed_items = [planetary_computer.sign(item).to_dict() for item in items]

# Define the pixel resolution for the final product
resolution = 10  # meters per pixel 
scale = resolution / 111320.0  # degrees per pixel for crs=4326 

data = stac_load(
    items,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    crs="EPSG:4326",  # Latitude-Longitude
    resolution=scale,  # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds,
    skip_broken=True
)

# [Plotting and other feature engineering code omitted for brevity]

# --- Feature Engineering: Compute Derived Indices ---

# Compute median composite.
median = data.median(dim="time").compute()

# Derived Indices:
ndvi = (median.B08 - median.B04) / (median.B08 + median.B04)  # NDVI
ndbi = (median.B11 - median.B08) / (median.B11 + median.B08)  # NDBI
ndwi = (median.B03 - median.B08) / (median.B03 + median.B08)  # NDWI
ndmi = (median.B08 - median.B11) / (median.B08 + median.B11)  # NDMI

# Texture measure using NDVI rolling variance (3x3 window)
ndvi_var = ndvi.rolling(latitude=3, longitude=3, center=True).var()

# Load Land Surface Temperature from separate Landsat file.
lst = rio.open_rasterio("Landsat_LST.tiff").squeeze()

# Helper function to remove conflicting coordinate.
def clean_da(da):
    return da.drop_vars('spatial_ref') if 'spatial_ref' in da.coords else da

ndvi_clean     = clean_da(ndvi)
ndbi_clean     = clean_da(ndbi)
ndwi_clean     = clean_da(ndwi)
ndmi_clean     = clean_da(ndmi)
ndvi_var_clean = clean_da(ndvi_var)
lst_clean      = clean_da(lst)

# Remove extra "band" dimension if present.
for da in [ndvi_clean, ndbi_clean, ndwi_clean, ndmi_clean, ndvi_var_clean, lst_clean]:
    if 'band' in da.dims:
        da = da.isel(band=0)
        
# Combine Derived Features into a single Dataset.
features = xr.Dataset({
    'NDVI': ndvi_clean,
    'NDBI': ndbi_clean,
    'NDWI': ndwi_clean,
    'NDMI': ndmi_clean,
    'NDVI_var': ndvi_var_clean,
    'LST': lst_clean
})
if 'spatial_ref' in features.coords:
    features = features.drop_vars('spatial_ref')

# Ensure each variable is strictly 2D (latitude, longitude).
new_vars = {}
for var in features.data_vars:
    da = features[var]
    extra_dims = [d for d in da.dims if d not in ['latitude', 'longitude']]
    if extra_dims:
        sel = {d: 0 for d in extra_dims}
        da = da.isel(**sel)
    new_vars[var] = da
features_clean = xr.Dataset(new_vars)

# ---- Export Feature DataFrame for Analysis ----
# Create a spatial subset and stack for DataFrame conversion.
features_subset = features_clean.isel(latitude=slice(0, 100), longitude=slice(0, 100))
features_stacked = features_subset.stack(pixel=("latitude", "longitude"))
df_features_full = features_stacked.to_dataframe().dropna()

# Remove any existing columns that conflict with index names.
for name in df_features_full.index.names:
    if name in df_features_full.columns:
        df_features_full = df_features_full.drop(columns=[name])


# Reset index so coordinate columns become normal columns
df_features_full = df_features_full.reset_index()

# Drop duplicate columns if they exist.
df_features_full = df_features_full.loc[:, ~df_features_full.columns.duplicated()]


sample_size = min(20000, len(df_features_full))
df_features = df_features_full.sample(n=sample_size, random_state=123)
print("Extracted feature DataFrame shape:", df_features.shape)
print(df_features.head())

# Save the features for further analysis.
df_features.to_csv("feature_engineering.csv", index=False)

# ---- Create GeoTIFF Output with Additional Bands ----
filename = "S2_sample_01.tiff"
# Pick a single time slice from the time series (e.g., time=7 → July 24, 2021)
data_slice = data.isel(time=7)

# Read the Submission_template.csv to compute full spatial extent.
sub_df = pd.read_csv("Submission_template.csv")
min_lon = sub_df['Longitude'].min()
max_lon = sub_df['Longitude'].max()
min_lat = sub_df['Latitude'].min()
max_lat = sub_df['Latitude'].max()
print("Computed submission bounds:", min_lon, min_lat, max_lon, max_lat)

# Get dimensions from the data_slice.
height = data_slice.dims["latitude"]
width = data_slice.dims["longitude"]

# Define the transformation using the submission bounds.
gt = rasterio.transform.from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
data_slice.rio.write_crs("epsg:4326", inplace=True)
data_slice.rio.write_transform(transform=gt, inplace=True)

# Create a GeoTIFF with 7 bands: B01, B04, B06, B08, B11, B03, and LST.
with rasterio.open(filename, 'w', driver='GTiff', width=width, height=height,
                   crs='epsg:4326', transform=gt, count=7, compress='lzw', dtype='float64') as dst:
    dst.write(data_slice.B01, 1)
    dst.write(data_slice.B04, 2)
    dst.write(data_slice.B06, 3)
    dst.write(data_slice.B08, 4)
    dst.write(data_slice.B11, 5)
    dst.write(data_slice.B03, 6)
    dst.write(lst_clean.values, 7)
    dst.close()

# You can use a shell command to list the new file:
# !dir *.tiff