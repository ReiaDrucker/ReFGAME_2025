import os
import lzma
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from awpy import DemoParser
from tqdm import tqdm
from multiprocessing import Pool
import contextlib
import io

# Function to read .xz archives from ESTA
def read_parsed_demo(filename):
    try:
        with lzma.LZMAFile(filename, "rb") as f:
            data = f.read()
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error in file {filename}: {e}")
                return None
    except (lzma.LZMAError, EOFError) as e:
        print(f"LZMA decompression error in file {filename}: {e}")
        return None

# Recursive key replacement function
def replace_key_recursive(obj, old_key, new_key):
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            updated_key = new_key if key == old_key else key
            new_obj[updated_key] = replace_key_recursive(value, old_key, new_key)
        return new_obj
    elif isinstance(obj, list):
        return [replace_key_recursive(item, old_key, new_key) for item in obj]
    else:
        return obj

# Function to parse and process a single ESTA file
def parse_and_process_file(file):
    base_path = '../demo-files/esta-main/data/lan'
    filepath = f'{base_path}/{file}'

    # Suppress stdout/stderr during this block
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            demo_parser = DemoParser()

            parsed_json = replace_key_recursive(read_parsed_demo(filepath), "matchId", "matchID")
            demo_parser.json = parsed_json

            parsed_demo_dict = demo_parser.parse_json_to_df()

            dfs = [x for x in parsed_demo_dict.values() if isinstance(x, pd.DataFrame)]
            extra_data = {k: v for k, v in parsed_demo_dict.items() if not isinstance(v, pd.DataFrame)}

            # Extract round info
            round_df = dfs[0]
            round_df['originFile'] = file

            # Extract player frame data and merge with player info
            st_pf_df = dfs[8]
            st_pf_df = st_pf_df[st_pf_df['isAlive'] == True].reset_index(drop=True)

            st_pf_df = st_pf_df[['roundNum', 'tick', 'seconds', 'side', 'name', 'team', 'x', 'y', 'z', 'velocityX', 'velocityY',
                  'velocityZ', 'viewX', 'viewY']]
            st_pf_df['originFile'] = file
            frames_and_player_frames_df = st_pf_df.merge(dfs[7], on=['roundNum', 'tick', 'seconds'], how='left')

            # Add extra data columns directly into merged df here
            for key in ['matchID', 'mapName', 'tickRate', 'playbackTicks']:
                frames_and_player_frames_df[key] = extra_data.get(key)

            return round_df, frames_and_player_frames_df

        except Exception:
            return None

if __name__ == '__main__':

    base_path = '../demo-files/esta-main/data/lan'
    files = os.listdir(base_path)

    # Reduce for testing
    # files = files[:10]

    rounds_path = '../Outputs/esta-lan-rounds-info.parquet'
    frames_path = '../Outputs/esta-lan.parquet'

    # Delete old files if they exist
    for path in [rounds_path, frames_path]:
        if os.path.exists(path):
            os.remove(path)

    # Initialize Parquet writers
    rounds_writer = None
    frames_writer = None

    print("BEGIN Processing")
    with Pool(processes=32) as pool:
        for result in tqdm(pool.imap_unordered(parse_and_process_file, files), total=len(files)):
            if result is None:
                continue

            round_df, st_pf_df = result

            if not round_df.empty:
                round_table = pa.Table.from_pandas(round_df)
                if rounds_writer is None:
                    rounds_writer = pq.ParquetWriter(rounds_path, round_table.schema)
                rounds_writer.write_table(round_table)

            if not st_pf_df.empty:
                frame_table = pa.Table.from_pandas(st_pf_df)
                if frames_writer is None:
                    frames_writer = pq.ParquetWriter(frames_path, frame_table.schema)
                frames_writer.write_table(frame_table)

    # Finalize Parquet files
    if rounds_writer:
        rounds_writer.close()
    if frames_writer:
        frames_writer.close()

    print("END Processing")
