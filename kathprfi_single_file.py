import datetime
import katdal
import numpy as np
import pandas as pd
from numba import jit, prange
from skimage.measure import block_reduce


def readfile(path):
    return katdal.open(path)


def extract_raw_flags(katds):
    """
    Extract raw flag data from a katdal dataset without loading visibilities.

    Parameters
    ----------
    katds : katdal.DataSet
        The dataset object.

    Returns
    -------
    dask.array.Array
        Lazy-selected raw flags.
    """
    info = katds.source.data.chunk_info["flags"]
    store = katds.source.data.store
    array_name = store.join(info["prefix"], "flags")

    full_raw_flags = store.get_dask_array(
        array_name,
        info["chunks"],
        info["dtype"],
        index=katds.source.data.preselect_index,
        errors=0,
    )

    keep = (katds._time_keep, katds._freq_keep, katds._corrprod_keep)

    selected_raw_flags = katdal.lazy_indexer.DaskLazyIndexer(full_raw_flags, keep)
    return selected_raw_flags.dataset


def config2dic(filepath):
    args_dict = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, val = line.split('=')
                args_dict[key.strip()] = val.strip()
    return args_dict


def remove_bad_ants(katdata):
    return [
        ant.name
        for ant in katdata.ants
        if 'stop' not in katdata.sensor[ant.name + '_activity']
    ]


def selection(katdata, pol_to_use, corrprod, scan, clean_ants, flag_type):
    katdata.select(
        corrprods=corrprod,
        pol=pol_to_use,
        scans=scan,
        ants=clean_ants,
        flags=flag_type
    )
    
    return


def NewFlagChunk(flag_chunk):
    return block_reduce(flag_chunk, block_size=(1, 8, 1), func=np.any)


def get_az_and_el(katdata):
    azmean = np.mean(katdata.az, axis=1) % 360
    elmean = np.mean(katdata.el, axis=1)
    return elmean, azmean

def get_time_idx(katdata):
    unix = katdata.timestamps
    seconds = unix % 86400
    # absolute 10-min bin since midnight
    time_bins = (seconds // 600).astype(np.int32)  
    return time_bins


def get_az_idx(azimuth, azbins):
    return np.clip(np.digitize(azimuth, azbins) - 1, 0, len(azbins) - 2)


def get_el_idx(elevation, elbins):
    return np.clip(np.digitize(elevation, elbins) - 1, 0, len(elbins) - 1)



def get_bl_idx(katdata, nant=64):
    """
    Map current dataset correlation products to indices in the full baseline grid.

    Returns indices aligned with katdata.corr_products.
    """

    # Full baseline list (reference grid: 2016) 
    A1, A2 = np.triu_indices(nant, 1)
    full_corr_products = np.array(
        [f"m{A1[i]:03d}m{A2[i]:03d}" for i in range(len(A1))]
    )

    # Create mapping: name → index
    full_map = {name: idx for idx, name in enumerate(full_corr_products)}

    # Get ONLY baselines present in dataset 
    corr_prods = np.array(
        [f"{a[:-1]}{b[:-1]}" for a, b in katdata.corr_products]
    )

    # Map to full indices ---
    bl_idx = np.array([full_map[cp] for cp in corr_prods], dtype=np.int32)

    return bl_idx


@jit(nopython=True, parallel=True)
def update_arrays(Time_idx, Bl_idx, El_idx, Az_idx, flag_chunk, master_z, counter_z):
    ntime_chunk = flag_chunk.shape[0]
    nfreq = flag_chunk.shape[1]
    nbl = len(Bl_idx)
    
    for j in range(ntime_chunk):
        t = Time_idx_all[i + j]
        el = El_idx_all[i + j]
        az = Az_idx_all[i + j]
    
        for bi in range(nbl):
            bl = Bl_idx[bi]
    
            # vectorized over frequency (IMPORTANT)
            vals = flag_chunk[j, :, bi]
    
            master_z[t, :, bl, el, az] += vals
            counter_z[t, :, bl, el, az] += 1

    return Master, Counter