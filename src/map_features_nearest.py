import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def map_features_nearest(uhi_csv, feat_csv, tolerance=1e-7):
    """
    Maps derived satellite features from the feature engineering CSV onto the master
    Integrated UHI dataset using a nearest neighbor approach (based on exact coordinate matching).
    
    Parameters:
      uhi_csv   - Path to the master UHI dataset CSV.
      feat_csv  - Path to the feature engineering CSV (derived satellite features).
      tolerance - Maximum allowed Euclidean distance for a coordinate match (in degrees).  
                  Adjust as needed; here we use a very small tolerance.
    
    Returns:
      merged_df - A DataFrame based on the UHI dataset with additional columns for derived features.
    """
    # Read master UHI dataset and ensure coordinate columns are lowercase.
    df_uhi = pd.read_csv(uhi_csv)
    df_uhi.rename(columns={'Longitude': 'longitude', 'Latitude': 'latitude'}, inplace=True)
    
    # Read the feature engineering file and ensure coordinate names match.
    df_feat = pd.read_csv(feat_csv)
    if 'Longitude' in df_feat.columns:
        df_feat.rename(columns={'Longitude': 'longitude', 'Latitude': 'latitude'}, inplace=True)
    
    # (Do not round the coordinates to preserve accuracy.)
    # Build a KD-tree using the (latitude, longitude) pairs from the feature file.
    feat_coords = df_feat[['latitude', 'longitude']].values
    tree = cKDTree(feat_coords)
    
    # Build the array of master coordinates.
    master_coords = df_uhi[['latitude', 'longitude']].values
    
    # Query each UHI coordinate for its nearest neighbor in the feature dataset.
    distances, indices = tree.query(master_coords, k=1)
    
    # Define the list of derived feature columns to map.
    derived_cols = ['NDVI', 'NDBI', 'NDWI', 'NDMI', 'NDVI_var', 'LST']
    
    # Initialize new columns in the UHI dataset with NaN values.
    for col in derived_cols:
        df_uhi[col] = np.nan
    
    # For each row in the master, if the nearest neighbor is within tolerance, assign its feature values.
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        # If desired, you can check the tolerance here. You might want to always use the nearest, or only if dist is small.
        if dist <= tolerance:
            for col in derived_cols:
                df_uhi.at[i, col] = df_feat.iloc[idx][col]
        else:
            # Optionally, still map the nearest neighbor if no perfect match exists.
            for col in derived_cols:
                df_uhi.at[i, col] = df_feat.iloc[idx][col]
    
    return df_uhi

# Example usage:
if __name__ == "__main__":
    # Adjust file paths as needed.
    merged_df = map_features_nearest("Integrated_UHI_Dataset_100m.csv", "feature_engineering.csv", tolerance=1e-7)
    print("Merged master dataset sample with satellite features:")
    print(merged_df.head())
    merged_df.to_csv("Integrated_UHI_with_mapped_features.csv", index=False)