import pandas as pd

# Load the CSV file
file_path = "data/Coords.csv"
df = pd.read_csv(file_path)

# Show the first few rows and column names to understand the structure
df.head(), df.columns
item_name = df['Item'].drop_duplicates().values

#print(f"item_name: {item_name}")

import numpy as np
from scipy.spatial import distance_matrix

# Define the radii to evaluate
radii = [100, 150, 200, 300]

# Store results in a dictionary
best_coords_per_radius = {}

# Extract coordinates as a NumPy array
coords = df[['X', 'Y']].values


# Compute the pairwise distance matrix
dist_matrix = distance_matrix(coords, coords)

for radius in radii:
    within_radius = (dist_matrix <= radius).sum(axis=1) - 1  # exclude self
    max_index = np.argmax(within_radius)
    best_coords_per_radius[radius] = {
        "Best Coordinate": ', '.join(map(str, tuple(coords[max_index]))),
        "Neighbor Count": str(within_radius[max_index])
    }

best_coords_per_radius
print(best_coords_per_radius)
#print(f"best_coordinate for {item_name} is {best_coordinate} , max_neighbors: {max_neighbors}")