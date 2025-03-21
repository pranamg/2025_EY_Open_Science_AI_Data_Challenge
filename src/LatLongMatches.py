import pandas as pd

# Read the two CSV files
fe = pd.read_csv("buffered_satellite_values_200m.csv")
sub = pd.read_csv("Training_data_uhi_index.csv")

# Standardize column names by stripping whitespace and converting to lower-case
fe.rename(columns=lambda x: x.strip().lower(), inplace=True)
sub.rename(columns=lambda x: x.strip().lower(), inplace=True)

# If necessary, select a unique pair of coordinates. 
# (In case the feature_engineering file contains duplicate coordinate columns, select the correct ones.)
# Here we assume the coordinate columns are named 'latitude' and 'longitude'
# and we round them to 9 decimals to match the other files.
fe_coords = set(zip(fe['latitude'].round(9), fe['longitude'].round(9)))
sub_coords = set(zip(sub['latitude'].round(9), sub['longitude'].round(9)))

# Compute similar (common) coordinate pairs and dissimilar (unique) ones
similar_coords = fe_coords.intersection(sub_coords)
union_coords = fe_coords.union(sub_coords)
dissimilar_coords = union_coords - similar_coords

# Print the counts
print("Total similar coordinate pairs:", len(similar_coords))
print("Total dissimilar coordinate pairs:", len(dissimilar_coords))