#!/usr/bin/env python3
import argparse
import ast
import logging
import os
#from matplotlib.pyplot import jet
import six
import time as tme
import time
import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import zarr
import kathprfi_single_file as kathp
import numba
from tqdm import tqdm
from numba import prange
import cProfile
from concurrent.futures import ProcessPoolExecutor, as_completed



start_time = time.time()
def initialize_logs():
    """
    Initialize the log settings
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.basicConfig(filename = 'kathprfi_logs.log', format='%(message)s', level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser(description='This package produces two 5-D arrays, '
            'which are the counter array and the master array.'
            'The arrays provides statistics about measured'
            'RFI from MeerKAT telescope.')
    #define the default values for the configuration
    DEFAULT_OUTPUT_DIR  = "/scratch/kvanqa/RFI_work/"
    DEFAULT_FILE_PATH = "/scratch/kvanqa/RFI_work"

    parser.add_argument('-b', '--bad', action='store',  type=str,
                        help='Path to save list of bad files')
    parser.add_argument('-g', '--good', action='store', type=str, default='\tmp',
                        help='Path to save bad files')
    parser.add_argument('-z', '--zarr', action='store', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='path to save output zarr file')
    parser.add_argument('--filename', action='store', type=str, default=DEFAULT_FILE_PATH,
                        help='Path to the CSV file')
    parser.add_argument('-p', '--pol', type=str, choices=['HH','HV','VH','VV'], help='polarization of interest')
    parser.add_argument('--band', type=str, choices=['L', 'U'], default='U', help='Band of interest (L or U)')
    parser.add_argument('-s','--scan', type=str, default='track', help='observation scan')
    parser.add_argument('--corrprod', type=str, default='cross', help='add correlation product')
    parser.add_argument('--flag_type', type=str, choices=['cal_rfi', 'ingest_rfi','data_lost','cam'], default='cal_rfi', help='flag type of interest')

    return parser
   
def main():
     # Initializing the log settings
    # Configuration dictionary directly in the script
    parser = create_parser()
    args = parser.parse_args()
    pol = args.pol
    band = args.band
    corrprod = args.corrprod
    scan = args.scan
    flag_type = args.flag_type
    filename_path = args.filename
    data = pd.read_csv('sci_Imaging_U_2025-03-01T00:00:00Z_2025-03-31T00:00:00Z.csv')
    Filename = data['FullLink'].values
    # Read in csv file with files to processq
    badfiles = []
    goodfiles = []
    initialize_logs()
    logging.info('MEERKAT HISTORICAL PROBABILITY OF RADIO FREQUENCY INTERFERENCE FRAMEWORK')
            
    for i in range(len(Filename)):
        #Initializing 5-D arrays
        master = np.zeros((24, 4096, 2016, 8, 24), dtype=np.uint16)
        counter = np.zeros((24, 4096, 2016, 8, 24), dtype=np.uint16)
        s = tme.time()

        logging.info('Adding file {} : {}'.format(i, Filename[i]))
        try:
            pathvis = Filename[i]
            vis = kathp.readfile(pathvis)
            logging.info('File number {} has been read'.format(i))
            #import pdb; pdb.set_trace()
            if len(vis.freqs) == 4096 or len(vis.freqs) == 32768:

                logging.info('Removing bad antennas')
                clean_ants = kathp.remove_bad_ants(vis)
                logging.info('Bad antennas has been removed.')
                good_flags = kathp.selection(vis, pol=pol, corrprod=corrprod, scan=scan,
                                                clean_ants=clean_ants, flag_type=flag_type )
                
                logging.info('Good flags has been returned')
                print(f"good_flags shape before slicing = {good_flags.shape}")

                if good_flags.shape[0] * good_flags.shape[1] * good_flags.shape[2] != 0:
                #if good_flags.size.compute() == 0:
                #    continue
                # Updating the array
                    ntime = good_flags.shape[0]
                    time_step = 5
                    if ntime <= time_step:
                        time_step = ntime
                    nant = 64
                    Bl_idx = kathp.get_bl_idx(vis, nant)
                    elbins = np.linspace(10, 80, 8)
                    azbins = np.arange(0, 360, 15)
                    el, az = kathp.get_az_and_el(vis)

                    
                    logging.info('Start to update the master and counter array')
                    sample_points = np.linspace(0, ntime - 1, num=10, dtype=int)
                    print(f"Selected sample points: {sample_points}")

                    for tm in sample_points:
                    #for tm in range(0, ntime, time_step):
                        time_slice = slice(tm, tm + time_step)
                        flag_chunk = good_flags[time_slice].astype(int)
                        print(f"Processing sample {tm} with time slice {time_slice}")
                    # for tm in range(0, ntime, time_step):
                    #     time_slice = slice(tm, tm + time_step)
                    #     print(f"Selected time slice: {time_slice}")
                    #     flag_chunk = good_flags[time_slice].astype(int) #.compute()
                        print(f"flag_chunk shape after slicing: {flag_chunk.shape}")

                        # average flags from 32k to 4k mode.
                        if good_flags.shape[1] == 32768 :
                            original_shape=flag_chunk.shape
                            flag_chunk = kathp.NewFlagChunk(flag_chunk)
                            downsample_factor = good_flags.shape[1] // 4096
                            vis_freqs_chunk = vis.freqs[::downsample_factor]
                        else:
                            original_shape = flag_chunk.shape
                            vis_freqs_chunk = vis.freqs
                            logging.info(f"Reduced flag_chunk from {original_shape} to {flag_chunk.shape}")
                        print("Full Time_idx array:", kathp.get_time_idx(vis))
                        print(f"ALL unique time indices before slicing: {np.unique(kathp.get_time_idx(vis))}")
                        #Time_idx = kathp.get_time_idx(vis)[time_slice]
                        Time_idx = kathp.get_time_idx(vis)[ time_slice]
                        print(f"Unique time indices for sample {tm}: {np.unique(Time_idx)}")
                        print(f"Unique time indices: {np.unique(Time_idx)}")
                        print(f"Unique Time_idx for sample {tm}: {np.unique(Time_idx)}")
                        El_idx = kathp.get_el_idx(el, elbins)[time_slice]
                        Az_idx = kathp.get_az_idx(az, azbins)[time_slice]

                        master, counter = kathp.update_arrays(Time_idx, Bl_idx, El_idx, Az_idx,
                                                                flag_chunk, master, counter)
                        print(f"Updated master and counter for sample {tm}")

                        # if master.shape[1] != flag_chunk.shape[1]:
                        #     raise ValueError("Mismatch in frequency dimension between master and flag_chunk")

                    logging.info('{} s has been taken to update file number {}'.format(i,
                                                                                        tme.time()
                                                                                        - s))
                    goodfiles.append(Filename[i])
                    logging.info('Creating Xarray Dataset')
                    ds = xr.Dataset({'master': (('time', 'frequency', 'baseline', 'elevation',
                                                    'azimuth'), master),
                    'counter': (('time', 'frequency', 'baseline', 'elevation', 'azimuth'), counter)},
                    {'time': np.arange(24), 'frequency': vis_freqs_chunk, 'baseline': np.arange(2016),
                        'elevation': np.linspace(10, 80, 8), 'azimuth': np.arange(0, 360, 15)})
                    logging.info('Saving dataset')

                    flname = os.path.join(os.getcwd(), f"{band}_{pol}_{Filename[i][46:56]}.zarr")
                    ds.to_zarr(flname, group='arr')
                    logging.info('Dataset has been saved')
                    print(f"Final master sum: {master.sum()}, Final counter sum: {counter.sum()}")

                else:
                    logging.info('{} selection has a problem'.format(Filename[i]))
                    badfiles.append(Filename[i])
                    pass
            else:
                logging.info('{} selection has a problem'.format(Filename[i]))
                badfiles.append(Filename[i])
                pass

            np.save(args.good,goodfiles)
            np.save(args.bad,badfiles)
            logging.info('File has been saved')
        

        except Exception as e:
            logging.info(e)
            continue

 
if __name__=="__main__":

    main()

    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process_file, i, Filename) for i in range(len(Filename))]
    #     for future in futures:
    #         future.result()
        # for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        #     pass 


#calculate the program's run Timetime
# end_time = time.time()
# print(f"program's runtime {(end_time - start_time)/60.}")


''' when running the script you simply parse the arguments in the following manner
ipython ipython github/kathprfi/script/kathprfi_tester.py -- -z . --filename sci_Imaging_U_2025-02-01T00:00:00Z_2024-02-28T00:00:00Z.csv -p 'HH' --band 'U' -s 'track' --corrprod 'cross' --flag_type 'ingest_rfi'
'''

         
       
