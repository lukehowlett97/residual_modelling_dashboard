# data_loader.py
import pandas as pd
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    if filename.endswith('.pkl'):
        # Remove the .pkl extension and split the filename
        parts = filename[:-4].split('_')
        if len(parts) >= 5:
            # The last part is system and prn, e.g., 'C06'
            system_prn = parts[-1]
            system = system_prn[0]
            prn = system_prn[1:]
            return {
                'filename': filename,
                'system': system,
                'prn': prn
            }
    return None

def list_years(data_folder, selected_folder):
    folder_path = data_folder / selected_folder
    if not folder_path.exists():
        return []
    years = [d.name for d in folder_path.iterdir() if d.is_dir()]
    years = [d for d in years if d[0] is not '_']
    return sorted(years)

def list_doys(data_folder, selected_folder, selected_years):
    doys = set()
    for year in selected_years:
        year_path = data_folder / selected_folder / year
        if year_path.is_dir():
            found_doys = [d.name for d in year_path.iterdir() if d.is_dir()]
            doys.update(found_doys)
    return sorted(doys)

def list_stations(data_folder, selected_folder, selected_years, selected_doys):
    stations = set()
    for year in selected_years:
        for doy in selected_doys:
            doy_path = data_folder / selected_folder / year / doy
            if doy_path.is_dir():
                found_stations = [d.name for d in doy_path.iterdir() if d.is_dir()]
                stations.update(found_stations)
    return sorted(stations)

def list_pkl_files(data_folder, selected_folder, selected_years, selected_doys, selected_stations):
    files = []
    for year in selected_years:
        for doy in selected_doys:
            for station in selected_stations:
                station_path = data_folder / selected_folder / year / doy / station
                if station_path.is_dir():
                    pkl_files = list(station_path.glob('*.pkl'))
                    for f in pkl_files:
                        file_info = parse_filename(f.name)
                        if file_info is None:
                            continue
                        file_info.update({
                            'filepath': str(f),
                            'year': year,
                            'doy': doy,
                            'station': station,
                            'system': file_info['system'],
                            'prn': file_info['prn']
                        })
                        files.append(file_info)
    return files

def load_data(file_paths):
    data_frames = {}
    for file_path in file_paths:
        file_path = Path(file_path)
        file_name = file_path.name
        df = pd.read_pickle(file_path)
        # Ensure 'epoch' is datetime
        if 'epoch' in df.columns:
            df['epoch'] = pd.to_datetime(df['epoch'])
        data_frames[file_name] = df
    return data_frames

def get_prns_by_system(files_info):
    prns_by_system = defaultdict(set)
    for f in files_info:
        system = f['system']
        prn = f['prn']
        prns_by_system[system].add(prn)
    # Convert sets to sorted lists
    prns_by_system = {s: sorted(list(prns)) for s, prns in prns_by_system.items()}
    return prns_by_system
