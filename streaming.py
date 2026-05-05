import numpy as np
import dask.array as da
import pandas as pd
import zarr
import xarray as xr
import time 
import logging
import kathprfi_single_file as kathp

from dask.diagnostics import ProgressBar


def get_selected_flags(vis):
    clean_ants = kathp.remove_bad_ants(vis)

    kathp.selection(
        vis,
        pol_to_use='HH',
        corrprod='cross',
        scan='track',
        clean_ants=clean_ants,
        flag_type=['cal_rfi', 'ingest_rfi']
    )

    return
    
def average_1d(arr, factor):
    if len(arr) % factor != 0:
        raise ValueError("Length must be divisible by factor")
    
    return arr.reshape(-1, factor).mean(axis=1)

def init_zarr_store(zarr_path, vis):
    """
    Initialize Zarr store with xarray using Dask arrays for lazy allocation.
    Much faster and memory-efficient.
    """
    ntime = 144
    nfreq = 4096
    nbl = 2016
    nel = 8
    naz = 24
    
    get_selected_flags(vis)
    # For 32k-4k averaging
    #freqs = average_1d(vis.freqs, 8)
    freqs = vis.freqs

    # Lazy arrays with zeros
    master = da.zeros((ntime, nfreq, nbl, nel, naz), dtype=np.uint16, chunks=(1, 512, 256, nel, naz))
    counter = da.zeros((ntime, nfreq, nbl, nel, naz), dtype=np.uint16, chunks=(1, 512, 256, nel, naz))

    coords = {
        "time":  np.arange(ntime),  # convert to seconds if desired
        "frequency": freqs[:nfreq],
        "baseline": np.arange(nbl),
        "elevation": np.linspace(10, 80, nel),
        "azimuth": np.arange(0, 360, 15),
    }

    ds = xr.Dataset(
        {
            "master": (("time", "frequency", "baseline", "elevation", "azimuth"), master),
            "counter": (("time", "frequency", "baseline", "elevation", "azimuth"), counter),
        },
        coords=coords
    )

    # Write to Zarr lazily (no full in-memory allocation)
    ds.to_zarr(zarr_path, mode="w", consolidated=False)

    return zarr_path


def stream_to_zarr(vis, zarr_path, correlator_mode="4k", target_mem_gb=4):
    """
    Adaptive streaming katdal → Zarr with per-scan processing.
    """

    logging.info("Preparing streaming pipeline...")

    ntime_raw, nfreq, nbl = vis.shape

    # Baselines
    nant = 64
    Bl_idx = kathp.get_bl_idx(vis, nant)

    # Elevation / azimuth bins
    nel = 8
    naz = 24
    elbins = np.linspace(10, 80, nel)
    azbins = np.arange(0, 360, 15)

    el, az = kathp.get_az_and_el(vis)
    El_idx_all = kathp.get_el_idx(el, elbins)
    Az_idx_all = kathp.get_az_idx(az, azbins)

    # Time bins
    Time_idx_all = kathp.get_time_idx(vis)

    # Open Zarr
    logging.info("Opening Zarr store")
    z = zarr.open(zarr_path, mode='r+')
    master_z = z['master']
    counter_z = z['counter']

    freq_idx = np.arange(nfreq)

    start_idx = 0
    scan_list = list(vis.scans())

    for i, (idx, state, target) in enumerate(vis.scans()):
        logging.info(f"Processing scan: {i+1} of {len(scan_list)}")

        flags_lazy = kathp.extract_raw_flags(vis)

        n_scan_times = vis.shape[0]
        end_idx = start_idx + n_scan_times

        ti_all = Time_idx_all[start_idx:end_idx]
        eli_all = El_idx_all[start_idx:end_idx]
        azi_all = Az_idx_all[start_idx:end_idx]

        # --- Estimate memory per time slice ---
        sample_flags = flags_lazy[0:1].compute().astype(np.uint8)
        bytes_per_time = sample_flags.nbytes

        target_bytes = target_mem_gb * 1024**3

        # factor 4 = flag + master + counter + overhead
        batch_size = max(1, int(target_bytes // (bytes_per_time * 4)))

        logging.info(f"Adaptive batch size: {batch_size} time steps")

        # --- Process scan in batches ---
        for t0 in range(0, n_scan_times, batch_size):
            t1 = min(t0 + batch_size, n_scan_times)

            with ProgressBar():
                scan_flags = flags_lazy[t0:t1].compute().astype(np.uint8)

            flag_chunk = (scan_flags != 0).astype(np.uint8)

            if correlator_mode == "32k":
                flag_chunk = kathp.NewFlagChunk(flag_chunk)

            ti = ti_all[t0:t1]
            eli = eli_all[t0:t1]
            azi = azi_all[t0:t1]

            # Expand indices (bounded!)
            ti_exp = ti[:, None, None]
            freq_exp = freq_idx[None, :, None]
            bl_exp = Bl_idx[None, None, :]
            eli_exp = eli[:, None, None]
            azi_exp = azi[:, None, None]

            start_time = time.time()

            # Read
            master_scan = master_z[ti_exp, freq_exp, bl_exp, eli_exp, azi_exp]
            counter_scan = counter_z[ti_exp, freq_exp, bl_exp, eli_exp, azi_exp]

            # Update
            master_scan += flag_chunk
            counter_scan += 1

            # Write back
            master_z[ti_exp, freq_exp, bl_exp, eli_exp, azi_exp] = master_scan
            counter_z[ti_exp, freq_exp, bl_exp, eli_exp, azi_exp] = counter_scan

            elapsed = time.time() - start_time

            logging.info(
                f"Batch {t0}:{t1} | size={flag_chunk.nbytes/1024**2:.2f} MB | "
                f"time={elapsed:.2f}s"
            )

        start_idx = end_idx