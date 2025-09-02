import tempfile
import os
import shutil
from contextlib import contextmanager
import numpy as np
import pandas as pd
import multiprocessing as mp
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from tqdm import tqdm
from scipy.spatial import KDTree, ConvexHull
from awpy.data import MAP_DATA
from scipy.optimize import curve_fit
from enum import Enum
import traceback
import gc


def round_to_nearest_value(data, val):
    return np.asarray([(e - e % val) if (e % val < (val / 2)) else (e + val - e % val) for e in data])

# Returns the centers of hypercubes resulting from dividing the unit hypercube divided into hypercubes of given side length
# Result of shape ((1/side_length))^n, n)
def generate_cube_centers(n, side_length):
    # Generate evenly spaced points along each axis
    axis_points = np.linspace(0.0, 1.0, int(1 / side_length) + 1)
    axis_centers = (axis_points[:-1] + axis_points[1:]) / 2  # Compute centers

    # Create a grid of centers in n-dimensional space
    grid = np.meshgrid(*([axis_centers] * n), indexing='ij')
    centers = np.stack(grid, axis=-1).reshape(-1, n)

    return centers


# For any points in dimension other than 2 or 3 consider if a PCA reduction to 2 dimensions is a good enough estimate
# Runtime will be on the order of how many dimensions you have
def fractal_dimension_box_counting(df, columns, box_sizes=None, box_center_KD_trees=None):
    """
    Calculate the fractal dimension of a set of points in n-dimensions using the box-counting method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    columns (list): List of column names in the DataFrame that contain the coordinates of the points.
    box_sizes (list): List of box sizes to use for the box-counting method. If None, a default range is used
    box_center_KD_trees (list): List of KD_trees with the centers of the boxes to use for the box-counting method. If None, a default range is used and built

    Returns:
    float: The fractal dimension of the set of points.
    """

    # Determine the range of box sizes if not provided
    if box_sizes is None:
        min_box_size = 1e-2
        max_box_size = 1.0
        num_box_sizes = 3
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num_box_sizes, base=10)

    # Extract the points from the DataFrame
    points = df[columns].values
    n_dimensions = points.shape[1]

    # Normalize points to a unit hypercube to reduce error caused by misaligned box-placement
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    range_coords = max_coords - min_coords

    # Add small epsilon only where the range is zero to remove divide by 0 errors in normalization
    epsilon = 1e-20
    range_coords[range_coords == 0] += epsilon
    points = (points - min_coords) / range_coords

    # Build a KDTree of the points for efficient range queries
    points_tree = KDTree(points)

    # Create the box_center_KD_trees if not done yet
    if box_center_KD_trees is None:
        box_center_KD_trees = []
        for box_size in box_sizes:
            # Generate the center of each box in n_dimensions and build a kd-tree out of them
            box_centers_tree = KDTree(generate_cube_centers(n_dimensions, box_size))
            box_center_KD_trees.append(box_centers_tree)

    box_counts = []

    # Count the number of non-empty boxes for each box size
    for i in range(len(box_sizes)):
        box_centers_tree = box_center_KD_trees[i]

        # Counts how many points are in each box by comparing the k-d trees
        # Result is a list of lists of the indices of all points within each box
        # Error induced by floating point precision can probably be reduced (but not done here)
        # Should calculate error due to float precision and add that to the radius as a tolerance (check the math on this if you implement it)
        points_within_boxes = box_centers_tree.query_ball_tree(points_tree, r=(box_sizes[i] / 2) * n_dimensions, p=1)

        # Count the number of non-empty boxes
        box_counts.append(len(points_within_boxes) - points_within_boxes.count([]))

    # Perform a linear regression in log-log to estimate the fractal dimension
    slope, intercept = np.polyfit(np.log(box_sizes), np.log(box_counts), 1)

    # Reduces some calculation error via the fact that -log(x) = log(1/x)
    fractal_dimension = -slope

    return fractal_dimension


# Assigns a unique ID to each row based on its position in a grid of increment size d
# Note that x,y,z values must be in range [-a, a]
def assign_cube_ids(df, max_coordinate_val, d, position_columns):
    df_new = df.copy()

    # Convert coordinates to grid indices
    for c in position_columns:
        df_new['grid_' + c] = ((df_new[c] + max_coordinate_val) // d).astype(int)

    # Create a unique grid loc (as a tuple)
    df_new['grid_loc_tuple'] = df_new[['grid_' + w for w in position_columns]].apply(tuple, axis=1)

    # Convert it to a single number
    unique_mapping = {}
    current_id = 1
    for id in np.asarray(df_new['grid_loc_tuple']):
        if id not in unique_mapping:
            unique_mapping[id] = current_id
            current_id += 1

    df_new['grid_loc_id'] = (df_new['grid_loc_tuple']).map(unique_mapping)

    return df_new


# Finds the shortest subsequence of S starting at position 'start' that has not appeared before, using a global set to track seen substrings.
# Sequence S must have a unique terminating character
def shortest_new_substring(S, start, global_seen):
    # Try substrings of increasing lengths
    for length in range(1, len(S) + 1 - start):
        substring = tuple(S[start:start + length])
        if substring not in global_seen:
            global_seen.add(substring)
            return length

    # Fallback: return max possible length if all substrings have been seen so far
    return len(S) - start


# Computes the entropy rate of a visit sequence S (numpy array of integers with unique terminating character).
def entropy_rate(symbol_sequence):
    # Return 0 if the sequence is empty
    if len(symbol_sequence) == 0:
        return 0

    # Should check for unique terminating character but we can assume it is done properly
    global_seen = set()

    # Compute the sum of shortest new substrings for each position 0 through L-1
    li_sum = sum(shortest_new_substring(symbol_sequence, i, global_seen) for i in range(len(symbol_sequence) - 1))

    return np.log(len(symbol_sequence)) * len(symbol_sequence) / li_sum


def entropy_func(x, c1, c2, c3, c4, c5):
    comb1 = c1 * pow(4 * pow(x[0], 2) * x[2], -1) * pow(x[1], 2)  # C1 needs to be non-negative
    comb2 = c2 * pow(4 * pow(x[0], 2) * x[2], -1)  # C2 needs to be non-negative
    comb3 = c3 * 2 * x[1] * pow(4 * pow(x[0], 2) * x[2], -1)  # C3 needs to be non-negative
    comb4 = c4 * x[1] * pow(x[0] * x[2], -1)  # C4 needs to be non-negative
    comb5 = c5 * pow(x[0] * x[2], -1)  # C5 needs to be non-negative
    comb6 = np.log(x[2])
    return pow(comb1 + comb2 + comb3 + comb4 + comb5, -1) * comb6


def calc_r_squared(x_data, y_data, variables):
    residuals = y_data - entropy_func(x_data, variables[0], variables[1], variables[2], variables[3], variables[4])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def calculate_scale_invariant_entropy_constants(df, columns, map, is_camera):

    if columns == ['x', 'y']:
        max_coordinate_val = int(np.max(np.abs([MAP_DATA.get(map).get('pos_x'), MAP_DATA.get(map).get('pos_y')])) + 1)
    elif columns == ['x', 'y', 'z']:
        # No pos_z in map data
        max_coordinate_val = int(np.max(np.abs([MAP_DATA.get(map).get('pos_x'), MAP_DATA.get(map).get('pos_y')])) + 1)
    elif columns == ['viewX', 'viewY'] and is_camera:
        max_coordinate_val = 360
    else:
        raise ValueError("Unsupported columns. Expected ['x', 'y'] , ['x', 'y', 'z'], or ['viewX', 'viewY']")



    grid_sizes = [1, 5, 10, 20, 50]
    sample_rates = [0.5, 1, 2, 4, 8]

    # Store per-round results
    all_results = []

    entropies = []
    string_lengths = []
    st_resolutions = []

    for d in grid_sizes:  # Cube side length
        for t in sample_rates:  # Sampling rate
            df_with_ids = assign_cube_ids(df, max_coordinate_val, d, columns)
            symbols = np.array(df_with_ids.grid_loc_id, dtype=np.int32)

            # Select every t*2-th element (since 0.5s is base)
            symbols = symbols[::int(t * 2)]
            symbols = np.append(symbols, 0)

            string_lengths.append(len(symbols))
            entropies.append(entropy_rate(symbols))
            st_resolutions.append((t, d))

    # Unzip tuple list into two separate lists
    sample_rates_unzipped, grid_sizes_unzipped = zip(*st_resolutions)

    # Store results per round
    all_results.append({
        'sample_rates': sample_rates_unzipped,
        'string_lengths': string_lengths,
        'grid_sizes': grid_sizes_unzipped,
        'entropies': entropies
    })

    for result in all_results:
        # input: T, D, L, independent variables and H as dependent variable
        inputs = np.column_stack((result['sample_rates'], result['grid_sizes'], result['string_lengths'])).T

        output = np.asarray(result['entropies'])

        fit_result = curve_fit(entropy_func, xdata=inputs, ydata=output, maxfev=50000, full_output=True, bounds=(0, np.inf))

        c_values = fit_result[0]

        r_squared = calc_r_squared(inputs, output, c_values)

        return fit_result, r_squared
    return None


def retry_on_fail(args, retries=1):
    for i in range(retries + 1):
        result = get_refgem(args)
        if result is not None:
            return result
        print(f'Retrying due to error on: {args}')
    return None

def normalized_convex_hull(points, map, is_camera, columns):


    if columns == ['x', 'y']:
        max_coordinate_val = int(np.max(np.abs([MAP_DATA.get(map).get('pos_x'), MAP_DATA.get(map).get('pos_y')])) + 1)
    elif columns == ['x', 'y', 'z']:
        # No pos_z in map data
        max_coordinate_val = int(np.max(np.abs([MAP_DATA.get(map).get('pos_x'), MAP_DATA.get(map).get('pos_y')])) + 1)
    elif columns == ['viewX', 'viewY'] and is_camera:
        max_coordinate_val = 360
    else:
        raise ValueError("Unsupported columns. Expected ['x', 'y'] , ['x', 'y', 'z'], or ['viewX', 'viewY']")

    # Scale all points to within [0,1] for each axis
    points = (points + max_coordinate_val) / (2 * max_coordinate_val)

    return ConvexHull(points, qhull_options='QJ')


def get_refgem(args):
    player_name, match_id, map_name, file_path, columns, is_camera, recovery_mapping, is_rounds = args
    if is_rounds:
        try:
            df = pd.read_parquet(file_path)
            round_results = []

            for round_num, round_df in df.groupby('roundNum'):

                # Check that there are at least 4 entries (need 3 to calculate convex hull)
                if round_df.size < 4:
                    print(f'Skippping {args[0]} {args[1]} {args[2]} round {round_num} due to insufficient number of rows')
                    round_results.append({
                        'roundNum': None,
                        'fractal_dim': None,
                        'ch_area': None,
                        'ch_volume': None,
                        'ch_area_normed': None,
                        'ch_volume_normed': None,
                        'fit_res': None,
                        'r_squared': None
                    })
                    continue

                try:
                    fractal_dim = fractal_dimension_box_counting(round_df, columns)
                except Exception as e:
                    print(f'Failed to compute fractal dimension for {args[0]} {args[1]} {args[2]} round {round_num}: {e}')
                    print(traceback.format_exc())
                    fractal_dim = None

                try:
                    ch = ConvexHull(round_df[columns].values, qhull_options='QJ')
                    ch_area = getattr(ch, 'area', None)
                    ch_volume = getattr(ch, 'volume', None)

                    # Convex Hull surface area and volume as a percentage of map size
                    ch_normed = normalized_convex_hull(round_df[columns].values, recovery_mapping['mapName'][map_name], is_camera, columns)
                    ch_area_normed = getattr(ch_normed, 'area', None)
                    ch_volume_normed = getattr(ch_normed, 'volume', None)
                except Exception as e:
                    print(f'Failed to compute ch for for {args[0]} {args[1]} {args[2]} round {round_num}: {e}')
                    print(traceback.format_exc())
                    ch_area = None
                    ch_volume = None
                    ch_area_normed = None
                    ch_volume_normed = None

                try:
                    fit_res, r_squared = calculate_scale_invariant_entropy_constants(
                        round_df, columns, recovery_mapping['mapName'][map_name], is_camera
                    )
                except Exception as e:
                    print(f'Failed to compute entropy_fit for {args[0]} {args[1]} {args[2]} round {round_num}: {e}')
                    print(traceback.format_exc())
                    fit_res = None
                    r_squared = None

                round_results.append({
                    'roundNum': round_num,
                    'fractal_dim': fractal_dim,
                    'ch_area': ch_area,
                    'ch_volume': ch_volume,
                    'ch_area_normed': ch_area_normed,
                    'ch_volume_normed': ch_volume_normed,
                    'fit_res': fit_res,
                    'r_squared': r_squared
                })

            return player_name, match_id, map_name, round_results
        except Exception as e:
            print(f"Error processing {args[0]} {args[1]} {args[2]}: {e}")
            return None
    else:
        try:
            df = pd.read_parquet(file_path)
            half_results = []

            # Only look at the first half of a match
            df = df.loc[df['roundNum'] <= 15]

            for side, side_df in df.groupby('side'):

                # Check that there are at least 4 entries (need 3 to calculate convex hull)
                if side_df.size < 4:
                    print(
                        f'Skippping {args[0]} {args[1]} {args[2]} side {side} due to insufficient number of rows')
                    half_results.append({
                        'side': None,
                        'fractal_dim': None,
                        'ch_area': None,
                        'ch_volume': None,
                        'ch_area_normed': None,
                        'ch_volume_normed': None,
                        'fit_res': None,
                        'r_squared': None
                    })
                    continue

                try:
                    fractal_dim = fractal_dimension_box_counting(side_df, columns)
                except Exception as e:
                    print(
                        f'Failed to compute fractal dimension for {args[0]} {args[1]} {args[2]} side {side}: {e}')
                    print(traceback.format_exc())
                    fractal_dim = None

                try:
                    ch = ConvexHull(side_df[columns].values, qhull_options='QJ')
                    ch_area = getattr(ch, 'area', None)
                    ch_volume = getattr(ch, 'volume', None)

                    # Convex Hull surface area and volume as a percentage of map size
                    ch_normed = normalized_convex_hull(side_df[columns].values, recovery_mapping['mapName'][map_name],
                                                       is_camera, columns)
                    ch_area_normed = getattr(ch_normed, 'area', None)
                    ch_volume_normed = getattr(ch_normed, 'volume', None)
                except Exception as e:
                    print(f'Failed to compute ch for for {args[0]} {args[1]} {args[2]} side {side}: {e}')
                    print(traceback.format_exc())
                    ch_area = None
                    ch_volume = None
                    ch_area_normed = None
                    ch_volume_normed = None

                try:
                    fit_res, r_squared = calculate_scale_invariant_entropy_constants(
                        side_df, columns, recovery_mapping['mapName'][map_name], is_camera
                    )
                except Exception as e:
                    print(f'Failed to compute entropy_fit for {args[0]} {args[1]} {args[2]} side {side}: {e}')
                    print(traceback.format_exc())
                    fit_res = None
                    r_squared = None

                half_results.append({
                    'side': side,
                    'fractal_dim': fractal_dim,
                    'ch_area': ch_area,
                    'ch_volume': ch_volume,
                    'ch_area_normed': ch_area_normed,
                    'ch_volume_normed': ch_volume_normed,
                    'fit_res': fit_res,
                    'r_squared': r_squared
                })

            return player_name, match_id, map_name, half_results
        except Exception as e:
            print(f"Error processing {args[0]} {args[1]} {args[2]}: {e}")
            return None



def process_refgem_result_to_df(result, is_rounds):

    if is_rounds:
        rows = []
        player_name, match_id, map_name, round_results = result

        for round_data in round_results:
            round_num = round_data['roundNum']
            area = round_data.get('ch_area')
            volume = round_data.get('ch_volume')
            frac_dim = round_data.get('fractal_dim')
            fit_res = round_data.get('fit_res')
            r_squared = round_data.get('r_squared')
            area_normed = round_data.get('ch_area_normed')
            volume_normed = round_data.get('ch_volume_normed')

            if (
                    fit_res is not None
                    and isinstance(fit_res, (list, tuple))
                    and len(fit_res) > 0
                    and isinstance(fit_res[0], (list, tuple, np.ndarray))
                    and len(fit_res[0]) >= 5
            ):
                C1, C2, C3, C4, C5, *_ = fit_res[0]
            else:
                C1 = C2 = C3 = C4 = C5 = None

            has_issue = any(v is None for v in [frac_dim, area, volume, fit_res, r_squared])

            rows.append({
                'name': player_name,
                'matchID': match_id,
                'mapName': map_name,
                'roundNum': round_num,
                'ch_area': area,
                'ch_volume': volume,
                'ch_area_normed': area_normed,
                'ch_volume_normed': volume_normed,
                'frac_dim': frac_dim,
                'C1': C1,
                'C2': C2,
                'C3': C3,
                'C4': C4,
                'C5': C5,
                'r_squared': r_squared,
                'has_issue': has_issue
            })

        return pd.DataFrame(rows)
    else:
        rows = []
        player_name, match_id, map_name, half_results = result

        for half_data in half_results:
            side = half_data['side']
            area = half_data.get('ch_area')
            volume = half_data.get('ch_volume')
            frac_dim = half_data.get('fractal_dim')
            fit_res = half_data.get('fit_res')
            r_squared = half_data.get('r_squared')
            area_normed = half_data.get('ch_area_normed')
            volume_normed = half_data.get('ch_volume_normed')

            if (
                    fit_res is not None
                    and isinstance(fit_res, (list, tuple))
                    and len(fit_res) > 0
                    and isinstance(fit_res[0], (list, tuple, np.ndarray))
                    and len(fit_res[0]) >= 5
            ):
                C1, C2, C3, C4, C5, *_ = fit_res[0]
            else:
                C1 = C2 = C3 = C4 = C5 = None

            has_issue = any(v is None for v in [frac_dim, area, volume, fit_res, r_squared])

            rows.append({
                'name': player_name,
                'matchID': match_id,
                'mapName': map_name,
                'side': side,
                'ch_area': area,
                'ch_volume': volume,
                'ch_area_normed': area_normed,
                'ch_volume_normed': volume_normed,
                'frac_dim': frac_dim,
                'C1': C1,
                'C2': C2,
                'C3': C3,
                'C4': C4,
                'C5': C5,
                'r_squared': r_squared,
                'has_issue': has_issue
            })

        return pd.DataFrame(rows)


# MODE CONFIGURATION
class Mode(Enum):
    CAMERA_ROUNDS = "camera_rounds"
    CAMERA_HALVES = "camera_halves"
    MODE_2D_ROUNDS = "2d_rounds"
    MODE_2D_HALVES = "2d_halves"
    MODE_3D_ROUNDS = "3d_rounds"
    MODE_3D_HALVES = "3d_halves"


def get_columns(mode: Mode):
    base_columns = {
        'camera': ['viewX', 'viewY'],
        '2d': ['x', 'y'],
        '3d': ['x', 'y', 'z']
    }

    base_mode = mode.value.split('_')[0]
    return base_columns[base_mode]


def get_output_paths(mode: Mode, testing: bool):
    # Extract base and detail (e.g., camera_rounds → camera, rounds)
    base_mode, detail = mode.value.split('_')
    suffix = f"{base_mode}-{detail}"  # e.g., camera-rounds

    base_name = f"refgem_summary-{suffix}"
    if testing:
        base_name += "-mini"

    return {
        "refgem": f"../Outputs/{base_name}.parquet",
        "input": "../Outputs/esta-lan.parquet"
    }


def compress_dataframe(df):
    # Downcast integer and float columns
    for col in df.select_dtypes(include='int').columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned' if (df[col] >= 0).all() else 'integer')
    for col in df.select_dtypes(include='float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert 'side' column to boolean with T = True and CT = False
    if 'side' in df.columns:
        df['side'] = df['side'].map({'T': True, 'CT': False}).astype('bool')

    # Encode object (string) columns as ints + make recovery maps
    str_cols_to_encode = ['name', 'mapName', 'matchID', 'originFile']
    encode_maps = {}

    for col in str_cols_to_encode:
        if col in df.columns:
            codes, uniques = pd.factorize(df[col], sort=True)
            df[col] = codes.astype(np.uint16 if len(uniques) < 65535 else np.uint32)
            encode_maps[col] = dict(enumerate(uniques))

    # Optionally drop unused columns
    drop_cols = [
        'ctTeamName', 'ctEqVal', 'ctAlivePlayers', 'ctUtility',
        'tTeamName', 'tEqVal', 'tAlivePlayers', 'tUtility',
        'tickRate', 'playbackTicks', 'seconds'
    ]

    df.drop(columns=drop_cols, inplace=True)

    return df, encode_maps

# Using this here due to the way our server is configured feel free to use any other temp directory of your choice
@contextmanager
def LocalTempDir(dirname="temp_parquets"):
    path = os.path.join(os.getcwd(), dirname)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)

def process_mode(mode, data, testing, recovery_mapping):
    print(f"===> Processing mode: {mode.value}")
    columns = get_columns(mode)
    base_mode, detail = mode.value.split('_')
    is_camera = (base_mode == 'camera')
    is_rounds = (detail == 'rounds')
    paths = get_output_paths(mode, testing)

    with LocalTempDir() as temp_dir:
        all_args = []
        for (name, match_id, map_name), df_grp in data.groupby(['name', 'matchID', 'mapName']):
            file_path = os.path.join(temp_dir, f"{name}_{match_id}_{map_name}.parquet")
            df_grp.to_parquet(file_path, index=False)
            all_args.append((name, match_id, map_name, file_path, columns, is_camera, recovery_mapping , is_rounds))

        total_ram_gb = psutil.virtual_memory().total / 1e9
        cpu_cores = mp.cpu_count()
        max_by_ram = int(total_ram_gb // 4.0)
        MAX_PROCESSES = max(1, min(cpu_cores, max_by_ram, 56))

        if os.path.exists(paths["refgem"]):
            os.remove(paths["refgem"])

        refgem_buffer = []
        refgem_writer = None

        print(f"Starting pool with {MAX_PROCESSES} processes.")
        with mp.Pool(processes=MAX_PROCESSES, maxtasksperchild=1) as pool:
            for result in tqdm(pool.imap_unordered(retry_on_fail, all_args, chunksize=1), total=len(all_args)):
                if result is not None:
                    refgem_buffer.append(process_refgem_result_to_df(result, is_rounds))
                    if len(refgem_buffer) >= 50:
                        refgem_df = pd.concat(refgem_buffer, ignore_index=True)
                        table = pa.Table.from_pandas(refgem_df, preserve_index=False)
                        if refgem_writer is None:
                            refgem_writer = pq.ParquetWriter(paths["refgem"], table.schema)
                        refgem_writer.write_table(table)
                        refgem_buffer.clear()
                        gc.collect()

        if refgem_buffer:
            refgem_df = pd.concat(refgem_buffer, ignore_index=True)
            table = pa.Table.from_pandas(refgem_df, preserve_index=False)
            if refgem_writer is None:
                refgem_writer = pq.ParquetWriter(paths["refgem"], table.schema)
            refgem_writer.write_table(table)

        if refgem_writer:
            refgem_writer.close()
        print(f"Completed mode: {mode.value}")

if __name__ == '__main__':
    testing = False
    modes = [
        Mode.CAMERA_ROUNDS,
        Mode.CAMERA_HALVES,
        Mode.MODE_3D_ROUNDS,
        Mode.MODE_3D_HALVES
    ]

    print('BEGIN Load Data')
    data = pd.read_parquet("../Outputs/esta-lan.parquet")
    if testing:
        data = data.loc[data.matchID == '046cdf91-97ab-4b8b-b19e-4c17ba3aa129']
        
    print('BEGIN Compression')
    data['tick'] = round_to_nearest_value(data['tick'], 64)
    data, recovery_mapping = compress_dataframe(data)

    rows = []
    for col, code_map in recovery_mapping.items():
        for code, original_val in code_map.items():
            rows.append({'column_name': col, 'code': code, 'original_value': original_val})
    encode_df = pd.DataFrame(rows)
    encode_df.to_parquet("../Outputs/recovery_mapping.parquet")

    for mode in modes:
        process_mode(mode, data, testing, recovery_mapping)