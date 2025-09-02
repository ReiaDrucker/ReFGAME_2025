import os
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp
from multiprocessing import shared_memory
from tqdm import tqdm

def round_to_nearest_value(data, val):
    return np.asarray([(e - e % val) if (e % val < (val / 2)) else (e + val - e % val) for e in data])

def find_journeys_and_dwells(df):
    if df.empty:
        return [], []

    journeys, dwells = [], []
    journey, dwell = [], []
    travelling = True

    for row in df.itertuples(index=True):
        currently_stopped = row.velocityX == 0 and row.velocityY == 0 and row.velocityZ == 0

        if not journeys and not dwells and not journey and not dwell:
            travelling = not currently_stopped

        if travelling and not currently_stopped:
            journey.append(row)

        elif travelling and currently_stopped:
            journey.append(row)
            journeys.append(journey)
            journey = []
            dwell.append(row)

        elif not travelling and currently_stopped:
            dwell.append(row)

        elif not travelling and not currently_stopped:
            dwells.append(dwell)
            dwell = []
            journey.append(row)

        travelling = not currently_stopped

    if journey:
        journeys.append(journey)
    if dwell:
        dwells.append(dwell)

    return journeys, dwells

def find_journeys_and_dwells_camera(df):
    if df.empty:
        return [], []

    journeys, dwells = [], []
    journey, dwell = [], []
    travelling = True
    last_row = None

    for row in df.itertuples(index=True):
        if last_row is None:
            currently_stopped = False
            travelling = True
        else:
            currently_stopped = row.viewX == last_row.viewX and row.viewY == last_row.viewY

        last_row = row

        if travelling and not currently_stopped:
            journey.append(row)

        elif travelling and currently_stopped:
            journey.append(row)
            journeys.append(journey)
            journey = []
            dwell.append(row)

        elif not travelling and currently_stopped:
            dwell.append(row)

        elif not travelling and not currently_stopped:
            dwells.append(dwell)
            dwell = []
            journey.append(row)

        travelling = not currently_stopped

    if journey:
        journeys.append(journey)
    if dwell:
        dwells.append(dwell)

    return journeys, dwells

def worker(args):
    # args = (group_df, is_camera, shm_j_name, shm_d_name, total_len)
    group_df, is_camera, shm_j_name, shm_d_name, total_len = args

    shm_j = shared_memory.SharedMemory(name=shm_j_name)
    shm_d = shared_memory.SharedMemory(name=shm_d_name)
    journey_arr = np.ndarray((total_len,), dtype=np.int32, buffer=shm_j.buf)
    dwell_arr = np.ndarray((total_len,), dtype=np.int32, buffer=shm_d.buf)

    journeys, dwells = (find_journeys_and_dwells_camera(group_df) if is_camera else find_journeys_and_dwells(group_df))

    # Use large base IDs to avoid collision between workers (optional)
    # Here we just assign incremental IDs per journey/dwell in this group
    journey_base = 0
    dwell_base = 0

    for journey in journeys:
        idxs = [r.Index for r in journey]
        for i in idxs:
            journey_arr[i] = journey_base
        journey_base += 1

    for dwell in dwells:
        idxs = [r.Index for r in dwell]
        for i in idxs:
            dwell_arr[i] = dwell_base
        dwell_base += 1

    shm_j.close()
    shm_d.close()
    return True

if __name__ == '__main__':
    mp.set_start_method('fork')

    for compute_mode in [True, False]:
        is_camera = compute_mode
        suffix = '-camera' if is_camera else ''

        print('Loading data...')
        data = pd.read_parquet('../Outputs/esta-lan.parquet')
        data['tick'] = round_to_nearest_value(data['tick'], 64)

        # Downcast numeric columns
        num_cols = data.select_dtypes(include=["number"]).columns
        data[num_cols] = data[num_cols].apply(pd.to_numeric, downcast='float').apply(pd.to_numeric, downcast='integer')

        total_len = len(data)
        print(f'Data length: {total_len}')

        # Create shared memory arrays for journey and dwell IDs, initialized to -1
        shm_journey = shared_memory.SharedMemory(create=True, size=total_len * 4)
        shm_dwell = shared_memory.SharedMemory(create=True, size=total_len * 4)
        journey_array = np.ndarray((total_len,), dtype=np.int32, buffer=shm_journey.buf)
        dwell_array = np.ndarray((total_len,), dtype=np.int32, buffer=shm_dwell.buf)
        journey_array.fill(-1)
        dwell_array.fill(-1)

        print('Preparing tasks...')
        group_keys = ['name', 'matchID', 'mapName', 'roundNum']
        groups = [(group_df.copy(), is_camera, shm_journey.name, shm_dwell.name, total_len) for _, group_df in data.groupby(group_keys)]

        MAX_PROCESSES = min(32, mp.cpu_count())
        print(f'Using up to {MAX_PROCESSES} worker processes.')

        BATCH_SIZE = 10
        with mp.Pool(processes=MAX_PROCESSES, maxtasksperchild=1) as pool:
            for i in tqdm(range(0, len(groups), BATCH_SIZE), desc='Batches'):
                batch = groups[i:i+BATCH_SIZE]
                pool.map(worker, batch)

        print('Attaching IDs to DataFrame...')
        data['journeyID' + suffix] = journey_array
        data['dwellID' + suffix] = dwell_array

        out_path = f'../Outputs/esta-annotated{suffix}.parquet'
        print(f'Writing output to {out_path}...')
        data.to_parquet(out_path, index=False)

        shm_journey.close()
        shm_journey.unlink()
        shm_dwell.close()
        shm_dwell.unlink()

        print('Done.')
