from numba import set_num_threads
set_num_threads(8)
import datetime
from datetime import timezone
import katdal
import dask
import numpy as np
import pandas as pd
from numba import jit
from numba import prange

from skimage.measure import block_reduce
import logging
def initialize_logs():
    """
    Initialize the log settings
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)

def readfile(path):
    """
<<<<<<< HEAD
    Read in the RDB file.

    Parameters
    ----------
    path : str
        RDB file

    Returns
    -------
    output_file : katdal.visdatav4.VisibilityDataV4
       katdal data object
    """
    vis = katdal.open(path)
    return vis


def remove_bad_ants(vis):
    """
    Extract good antennas by checking antenna activity list.

    Parameters
    ----------
    vis : katdal.visdatav4.VisibilityDataV4
        katdal data object

    Returns
    ---------
    output : python list
          List of good antennas
    """
    # This pull all the antenna used for observation
    AntList = []
    for ant in vis.ants:
        AntList.append(ant.name)

    # This will give the antenna activity list
    AntsActivity = []
    for AntName in AntList:
        AntsActivity.append((AntName, vis.sensor[AntName+'_activity']))
    # Check if there is a stop state in antenna activity list and if true remove
    # the antenna
    for i in range(len(AntsActivity)):
        if 'stop' in AntsActivity[i][1]:
            AntList.remove(AntsActivity[i][0])
        else:
            pass
    return AntList

def stored_flags(vis):
    data = vis.source.data
    info = data.chunk_info["flags"]
    array_name = data.store.join(info["prefix"], "flags")
    return data.store.get_dask_array(
        array_name,
        info["chunks"],
        info["dtype"],
        index=data.preselect_index,
        errors=katdal.flags.CAL_RFI  # manual change this accordingly
    )

def selection(vis, pol, corrprod, scan, clean_ants, flag_type):
    """
    Do subselection of the dataset based on the given parameters.
    
    In addition, add target tags to ensure that the selected data is valid and has aleast calibration information
    We do that by intersecting vis targets-tags with good known good-tags 

    Parameters:
    -----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object
    pol_to_use : str
        polarization product to use [HH or VV or VH or HV] default: 'HH'
    corrprod : str
        type of correlation product, either 'cross' or 'auto'. default: 'cross'
    scan : str
        type of a scan to use [track or slew]. default: 'track'
    clean_ants : python list
        list of clean antennas
    flag_type : python list
        type of flag/s [cal_rfi, ingest_rfi, cam]. default: cal_rfi

    Returns:
    --------
    output : katdal.lazy_indexer.DaskLazyIndexer
        sub selected katdal lazy indexer of RFI flags
    """
    initialize_logs()
    logging.info('checks cal report')
    good_tags = set(["target", "bpcal", "delaycal","fluxcal","gaincal","polcal"])
    good_targets = []
    target = vis.catalogue.targets[vis.target_indices[1]]
    for tar in vis.target_indices:
        target = vis.catalogue.targets[tar]
        if len(good_tags.intersection(target.tags))>0:
                good_targets.append(target)
    
    vis.select(corrprods=corrprod, pol=pol, scans=scan, ants=clean_ants,flags=flag_type, targets = good_targets)
    #dask.config.set(num_workers=16)
    #flag = stored_flags(vis)
    flag = vis.flags
    #subset_flags = flag[:int(0.75*flag.shape[0]), :, :int(0.75*flag.shape[2])]
    logging.info(f"observation has tags {set(good_targets)}")
    return flag #.compute()

def NewFlagChunk(flag_chunk):
    """
    Reduce 32k flag array to 4k flag array.
    
    NB: If any of the averaged samples is TRUE then
        the average becomes true.
    
    Paramaters:
    -----------
    flag : numpy array
        flags array
        
    Returns:
    --------
    output : numpy array
        average numpy array
    """
    averaged_chunk = block_reduce(flag_chunk, block_size=(1, 8, 1), func=np.any)
    return averaged_chunk 


def get_az_and_el(vis):
    """
    Get the telescope pointings (i.e. elevation and azimuth).

    Parameters:
    -----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object

    Returns:
    --------
    output: numpy arrays
        numpy arrays of avaraged elevation and azimuth of all antennas per each time stamp
    """
    # Getting the azmuth and elevation
    azmean = np.mean(vis.az, axis=1) % 360
    elmean = np.mean(vis.el, axis=1)
    return elmean, azmean

def get_time_idx(vis):

    """
    Convert unix time to hour of a day.

    Parameters:
    -----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object

    Returns:
    --------
    output: numpy array
       numpy array with time dumps converted to hour of a day
    """
    unix = vis.timestamps
    hour = []
    for i in range(len(unix)):
        # Convert Unix timestamp to UTC hour
        utc_time = datetime.datetime.utcfromtimestamp(unix[i]).strftime('%H')
        if utc_time == '24':
            hour.append(0)
        else:
            hour.append(int(utc_time))

    return np.array(hour, dtype=int)

def get_az_idx(azimuth, azbins):
    """
    Get the azimuth angle indices.

    Parameters:
    -----------
    azimuth : numpy array
           array of Azimuthal angle
    azbins : numpy array
         array of azimuthal bins

    Returns:
    --------
    output : numpy array
     array of elevation indices
    """
    az_idx = []
    for az in azimuth:
        for j in range(len(azbins)-1):
            if azbins[j] <= az < azbins[j+1]:
                az_idx.append(j)
    return np.array(az_idx)


def get_el_idx(elevation, elbins):
    """
    Get the elevation angle indices.

    Parameters:
    -----------
    elevation : numpy array
        array of elevation angles
    azbins : numpy array
        array of elevation bins

    output : numpy array
       array of elevation indices
    """
    el_idx = []

    for el in elevation:
        for j in range(len(elbins)):
            if elbins[j] <= el < elbins[j]+10:
                el_idx.append(j)
    return np.array(el_idx, dtype=np.int32)


def get_corrprods(vis):
    """
    Get the correlation products

    Parameters:
    ----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object

    Returns:
    --------
    output : numpy array
         array of correlation products
    """
    bl = vis.corr_products
    bl_idx = []
    for i in range(len(bl)):
        bl_idx.append((bl[i][0][0:-1]+bl[i][1][0:-1]))
    return np.array(bl_idx)


def get_bl_idx(vis, nant):
    """
    Get the indices of the correlation products.

    Parameters:
    -----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object
    nant : int
       number of antennas

    Returns:
    --------
    output : numpy array
       array of baseline indices
    """
    nant = nant
    A1, A2 = np.triu_indices(nant, 1)
    # Creating baseline antenna combinations
    corr_products = np.array(['m{:03d}m{:03d}'.format(A1[i], A2[i]) for i in range(len(A1))])
    df = pd.DataFrame(data=np.arange(len(A1)), index=corr_products).T
    corr_prods = get_corrprods(vis)
    bl_idx = df[corr_prods].values[0].astype(np.int32)
    return bl_idx


@jit(nopython=True, parallel=True)
def update_arrays(Time_idx, Bl_idx, El_idx, Az_idx, Good_flags, Master, Counter):
    """
    Update the master and counter array

    Parameters:
    -----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object
    azbins : numpy array
        array of azimuthal bins
    elbins : numpy array
        array of elevation bins
    Good_flags : katdal.lazy_indexer.DaskLazyIndexer
        sub selected katdal lazy indexer of RFI flags
    Master : numpy array
       array contains the number of RFI points per voxel with dimension of [T, F, B, Az, El]
    Counter : numpy array
      array contains the total number of observations per voxel with dimension of [T, F, B, Az, El]

    Returns:
    -------
    output : numpy array
      updated master and counter array
    """
   
    cstep = 128
    cblocks = (4096 + cstep - 1) // cstep
    for cblock in prange(cblocks):
        c_start = cblock * cstep
        c_end = min(4096, c_start + cstep)
        for k in range(c_start, c_end):
            for i in range(len(Bl_idx)):
                for j in range(len(Time_idx)):
                    Master[Time_idx[j], k, Bl_idx[i], El_idx[j], Az_idx[j]] += Good_flags[j, k, i]
                    Counter[Time_idx[j], k, Bl_idx[i], El_idx[j], Az_idx[j]] += 1
    return Master, Counter
