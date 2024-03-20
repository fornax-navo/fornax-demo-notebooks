import os
os.environ["OMP_NUM_THREADS"] = "2"  # restrict numpy multithreading. do before importing numpy.

import multiprocessing as mp
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd


DIR_OUT = Path(__file__).parent / "output"


def get_gaia_lightcurves(num_sample=100, verbose=False):
    lightcurves = []
    for i in range(num_sample):
        dim = 100
        matrix = np.random.rand(dim, dim)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)  # 19ms
        lstsq = np.linalg.lstsq(matrix, matrix, rcond=None)  # 9.5ms
        lightcurves.append(
            (dim, num_sample, eigenvalues.sum().real, eigenvectors.sum().real, lstsq[0].sum().real)
        )
        if verbose:
            print("done", i, end="\r", flush=True)
        sleep(0.1)
    lightcurves_df = _to_df(lightcurves, "gaia", num_sample)
    return lightcurves_df


def get_heasarc_lightcurves(num_sample=100, verbose=False):
    lightcurves = []
    for i in range(num_sample):
        dim = 10
        matrix = np.random.rand(dim, dim)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        lstsq = np.linalg.lstsq(matrix, matrix, rcond=None)
        lightcurves.append(
            (dim, num_sample, eigenvalues.sum().real, eigenvectors.sum().real, lstsq[0].sum().real)
        )
        if verbose:
            print("done", i, end="\r", flush=True)
        sleep(0.05)
    lightcurves_df = _to_df(lightcurves, "heasarc", num_sample)
    return lightcurves_df


def get_ztf_lightcurves(num_sample=100, verbose=False, nworkers=6):
    sleep(min(3600, 0.1 * num_sample))
    num_one_sample = min(num_sample // nworkers + 1, 100)
    args = [num_one_sample for _ in range(num_sample // num_one_sample + 1)]
    chunksize = min(len(args) // nworkers, 100)
    with mp.Pool(nworkers) as pool:
        lightcurves = []
        for i, results in enumerate(pool.imap_unordered(_ztf_run_one, args, chunksize=chunksize)):
            lightcurves.extend(results)
            if verbose:
                print("done", i, end="\r", flush=True)

    lightcurves_df = _to_df(lightcurves, "ztf", num_sample)
    size = (len(lightcurves_df.index) * 100, len(lightcurves_df.columns))
    random_df = pd.DataFrame(data=np.random.uniform(10, 20, size=size), columns=lightcurves_df.columns)
    lightcurves_df = pd.concat([lightcurves_df, random_df])
    return lightcurves_df


def get_wise_lightcurves(num_sample=100, verbose=False):
    lightcurves = []
    for i in range(num_sample):
        dim = 500
        matrix = np.random.rand(dim, dim)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)  # 19ms
        inverse = np.linalg.inv(matrix)  # 0.84ms
        lstsq = np.linalg.lstsq(matrix, inverse, rcond=None)  # 10.8ms
        lstsq_self = np.linalg.lstsq(matrix, matrix, rcond=None)  # 9.5ms
        svd = np.linalg.svd(matrix)  # 6.12ms
        svd_inverse = np.linalg.svd(inverse)  # 6.12ms
        qr = np.linalg.qr(matrix)  # 1.75ms
        qr_inverse = np.linalg.qr(inverse)  # 1.75ms
        power = np.linalg.matrix_power(matrix, 100)  # 0.964ms
        inner = np.inner(matrix, inverse)  # 0.114ms
        lightcurves.append(
            (dim, num_sample, eigenvalues.sum().real, eigenvectors.sum().real, lstsq[0].sum().real)
        )
        if verbose:
            print("done", i, end="\r", flush=True)
        sleep(0.05)
    lightcurves_df = _to_df(lightcurves, "wise", num_sample)
    return lightcurves_df


def _ztf_run_one(num_sample):
    lightcurves = []
    for i in range(num_sample * 10):
        dim = 100
        matrix = np.random.rand(dim, dim)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)  # 19ms
        inverse = np.linalg.inv(matrix)  # 0.84ms
        lstsq = np.linalg.lstsq(matrix, inverse, rcond=None)  # 10.8ms
        svd = np.linalg.svd(matrix)  # 6.12ms
        qr = np.linalg.qr(matrix)  # 1.75ms
        power = np.linalg.matrix_power(matrix, 100)  # 0.964ms
        inner = np.inner(matrix, inverse)  # 0.114ms
        lightcurves.append(
            (dim, num_sample, eigenvalues.sum().real, eigenvectors.sum().real, lstsq[0].sum().real)
        )
        sleep(1)
    return lightcurves


def _to_df(lightcurves, archive, num_sample):
    my_dir_out = DIR_OUT / f"num_sample={num_sample}"
    my_dir_out.mkdir(exist_ok=True, parents=True)

    columns = ["dim", "num_sample", "eigenvals_sum", "eigenvecs_sum", "lstsq_sum"]
    lightcurves_df = pd.DataFrame(lightcurves, columns=columns)

    lightcurves_df.to_parquet(f"{my_dir_out}/{archive}.snappy.parquet")
    return lightcurves_df
