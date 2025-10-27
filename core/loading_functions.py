import sys,os,os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor


sys.path.append("../")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
# you're going to have to reconfigure this to point to your IC (sorry!)
os.environ['ICTDIR']='/home/e78368jw/Documents/NEXT_CODE/IC'

from invisible_cities.io.dst_io             import load_dst, load_dsts, df_writer


from functools import partial

def load_single_file(file_path, group, node):
    '''
    Load data from a single h5 file and produce dataframes for /group/node

    Args:
        file_path       :       str
                                Path to the h5 file to be loaded.

    Returns:
        tracks_df       :       pandas.DataFrame
                                DataFrame containing the /group/node data.
    '''
    try: 
        tracks_df = load_dst(file_path, group, node)
        return tracks_df
    except Exception as e:
        print(f'File {file_path} broke with error:\n{e}', flush = True)
        x = pd.DataFrame()
        return x

## FUNCTIONS ##

def load_data_fast(folder_path, group, node):
    '''
    Load multiple h5 files and produce concatenated dataframes for /group/node, /group/node, and their corresponding eventmap.

    Args:
        folder_path     :       str
                                Path to the folder containing the h5 files.

    Returns:
        tracks          :       pandas.DataFrame
                                Concatenated DataFrame containing the /group/node data from all h5 files.
        
        particles       :       pandas.DataFrame
                                Concatenated DataFrame containing the /group/node data from all h5 files, with the 'event_id' column modified.

        eventmap        :       pandas.DataFrame
                                Concatenated DataFrame containing the event map from all h5 files.
    '''
    
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    file_paths = [os.path.join(folder_path, f) for f in file_names]

    load_files = partial(load_single_file, group = group, node = node)

    # Use ProcessPoolExecutor to parallelize the data loading process
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(load_files, file_paths))
    
    # Separate the results into respective lists
    tracks_list = results

    # Concatenate all the dataframes at once
    tracks = pd.concat(tracks_list, axis=0, ignore_index=True)

    return tracks