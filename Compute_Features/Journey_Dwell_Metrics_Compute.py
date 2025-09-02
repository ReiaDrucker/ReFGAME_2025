import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance


def calculate_metrics_generic(s, coord_cols):
    s = s.sort_values(by='tick')
    coords = s[coord_cols].to_numpy()
    step_displacements = [distance.euclidean(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]

    # Detect the correct ID column (e.g., dwellID, dwellID-camera, journeyID-camera)
    id_col = next((col for col in s.columns if col.startswith('dwellID') or col.startswith('journeyID')), None)
    if id_col is None:
        raise ValueError("No dwellID or journeyID column found in group.")

    return pd.Series({
        'name': s.name.iloc[0],
        'matchID': s.matchID.iloc[0],
        'mapName': s.mapName.iloc[0],
        'roundNum': s.roundNum.iloc[0],
        id_col: s[id_col].iloc[0],
        'duration': s['tick'].iloc[-1] - s['tick'].iloc[0],
        'total_displacement': distance.euclidean(coords[0], coords[-1]),
        'step_displacements': step_displacements
    })


if __name__ == '__main__':
    tqdm.pandas()

    for compute_mode in [True, False]:
        isCamera = compute_mode
        suffix = '-camera' if isCamera else ''
        coord_cols = ['viewX', 'viewY'] if isCamera else ['x', 'y', 'z']

        # Load and split into dwell/journey groups
        df = pd.read_parquet(f'../Outputs/esta-annotated{suffix}.parquet')

        dwell_id_col = f'dwellID{suffix}'
        journey_id_col = f'journeyID{suffix}'

        d_df = df.loc[df[dwell_id_col] != -1].drop(columns=journey_id_col)
        j_df = df.loc[df[journey_id_col] != -1].drop(columns=dwell_id_col)

        # Group and compute metrics
        dwell_metrics_df = d_df.groupby(
            ['name', 'matchID', 'mapName', 'roundNum', dwell_id_col]
        ).progress_apply(lambda s: calculate_metrics_generic(s, coord_cols)).reset_index(drop=True)

        journey_metrics_df = j_df.groupby(
            ['name', 'matchID', 'mapName', 'roundNum', journey_id_col]
        ).progress_apply(lambda s: calculate_metrics_generic(s, coord_cols)).reset_index(drop=True)

        # Save all outputs
        d_df.to_parquet(f'../Outputs/all_dwells{suffix}.parquet', index=False)
        j_df.to_parquet(f'../Outputs/all_journeys{suffix}.parquet', index=False)
        dwell_metrics_df.to_parquet(f'../Outputs/dwell_metrics{suffix}.parquet', index=False)
        journey_metrics_df.to_parquet(f'../Outputs/journey_metrics{suffix}.parquet', index=False)
