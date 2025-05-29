import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

def load_data(file_path: str) -> pd.DataFrame:
    """Load TSV file into a DataFrame."""
    df = pd.read_csv(file_path, sep='\t', header=None, index_col=False).rename(columns={0: 'Item', 1: 'Dummy', 2: 'X', 3: 'Y'})
    df = df.drop('Dummy', axis=1)
    df.drop_duplicates(inplace=True)
    return df

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    return distance_matrix(coords, coords)

def find_best_coords_within_radius(df: pd.DataFrame, coords: np.ndarray, radii: list) -> dict:
    """Find the coordinate with the most neighbors within each radius."""
    results = {}
    dist_matrix = compute_distance_matrix(coords)

    for radius in radii:
        within_radius = (dist_matrix <= radius).sum(axis=1) - 1  # exclude self
        max_index = np.argmax(within_radius)
        best_coord = coords[max_index]

        closest_points = get_closest_coords(df.copy(), best_coord, radius)

        results[radius] = {
            "Best Coordinate": ', '.join(map(str, tuple(coords[max_index]))),#tuple(best_coord),
            "Neighbor Count": int(within_radius[max_index]),
            "Closest Coords": closest_points.to_string(index=False)
        }

    return results

def get_closest_coords(df: pd.DataFrame, fixed_point: tuple, radius: float) -> pd.DataFrame:
    """Return coordinates within a given radius of the fixed point."""
    distances = np.linalg.norm(df[['X','Y']].values - np.array(fixed_point), axis=1)
    df = df.copy()
    df['Distance'] = distances.round()
    return df[df['Distance'] <= radius].sort_values('Distance')

def print_closest_coords(radius, result):
    print(f"\nRadius: {radius}")
    print(f"Best Coordinate: {result['Best Coordinate']}")
    print(f"Neighbor Count: {result['Neighbor Count']}")
    print("Closest Coords:")
    print(result['Closest Coords'])

def main():
    file_path = "data/coords.tsv"
    radii = [100, 150, 200]
    max_radii = max(radii)    
    df = load_data(file_path)
    coords = df[['X','Y']].values

    best_coords = find_best_coords_within_radius(df, coords, radii)
    for radius, result in best_coords.items():
        print_closest_coords(radius, result)

    # Example: get closest points to a fixed coordinate
    fixed_point = (771, 963)
    closest_coords = get_closest_coords(df, fixed_point, max_radii)
    #print(f"\nClosest to fixed point ({fixed_point[0]}, {fixed_point[1]}) within a radius of {max_radii}, Neighbor count {len(closest_coords)}: \n{closest_coords.to_string(index=False)}")
    
if __name__ == "__main__":
    main()
