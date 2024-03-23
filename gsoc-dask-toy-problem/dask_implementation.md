---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
---

```{code-cell}
# Dask implementation

import archives
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from functools import partial

# Function to execute and convert pandas dataframe to dask dataframe.
def execute_and_convert(func, num_samples):
    pdf = func(num_samples)
    # Converting Pandas DataFrame to Dask DataFrame.
    ddf = dd.from_pandas(pdf, npartitions=8)  # we can adjust npartitions as needed.
    return ddf


# This function utilizes Dask to parallelize the execution of multiple lightcurve retrieval functions with a specified number of samples, gathers the results, and returns them as pandas dataframe.

def parallelize_lightcurves(num_samples):
    # starting a localCluster and client.
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, processes=False)
    client = Client(cluster)
    
    # Partial functions for each lightcurve retrieval function.
    gaia_task = partial(execute_and_convert, archives.get_gaia_lightcurves, num_samples)
    heasarc_task = partial(execute_and_convert, archives.get_heasarc_lightcurves, num_samples)
    wise_task = partial(execute_and_convert, archives.get_wise_lightcurves, num_samples)
    ztf_task = partial(execute_and_convert, archives.get_ztf_lightcurves, num_samples)
   
    future_results = client.compute([gaia_task(), heasarc_task(), wise_task(), ztf_task()])

    # Gathering results. At this point, 'results' contains the Dask DataFrames for each of the functions.
    results = client.gather(future_results)

    # Concatenating the Dask DataFrames into Pandas DataFrame.
    concatenated_df = pd.concat(results)
    client.shutdown()
    return concatenated_df

num_samples = 5
results = parallelize_lightcurves(num_samples)
```
