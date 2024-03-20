---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
# IF needed to install required libraries
#!pip install numpy pandas datetime matplotlib dask distributed fastparquet pyarrow
```

# Execute archive.py functions serially

```{code-cell} ipython3
import archives
from dask.distributed import Client, as_completed, LocalCluster
import pandas as pd
import dask.dataframe as dd
import time
```

```{code-cell} ipython3
def serial_lightcurves(num_sample):
    start_time = time.time() 
    functions = [archives.get_gaia_lightcurves,archives.get_heasarc_lightcurves,archives.get_wise_lightcurves,archives.get_ztf_lightcurves]
    gaia_df = archives.get_gaia_lightcurves(num_sample)
    heasarc_df = archives.get_heasarc_lightcurves(num_sample)
    wise_df = archives.get_wise_lightcurves(num_sample)
    ztf_df = archives.get_ztf_lightcurves(num_sample)
    lightcurves_df = pd.concat([gaia_df, heasarc_df, wise_df, ztf_df])
    end_time = time.time()
    execution_time = end_time - start_time
    return lightcurves_df, execution_time
```

```{code-cell} ipython3
lightcurves_df, exe_time = serial_lightcurves(5)
print("Time for Serial execution with num_samples = 5: ", exe_time)
lightcurves_df, exe_time = serial_lightcurves(100)
print("Time for Serial execution with num_samples = 100: ", exe_time)
```

# Execute archive.py functions in parallel

```{code-cell} ipython3
def parallel_lightcurves(num_sample):
    local_cluster = LocalCluster(processes=False)
    # Start cluster
    client = Client(local_cluster)
    start_time = time.time()
    functions = [archives.get_gaia_lightcurves,archives.get_heasarc_lightcurves,archives.get_wise_lightcurves,archives.get_ztf_lightcurves]
    tasks = []
    for func in functions:
        tasks.append(client.submit(func, num_sample))
    lightcurves_df = pd.concat([task.result() for task in tasks])
    end_time = time.time()
    execution_time = end_time - start_time
    client.close()
    local_cluster.close()
    return lightcurves_df, execution_time
```

```{code-cell} ipython3
lightcurves_df, exe_time = parallel_lightcurves(5)
print("Time for Serial execution with num_samples = 5: ", exe_time)
lightcurves_df, exe_time = parallel_lightcurves(100)
print("Time for Serial execution with num_samples = 100: ", exe_time)
```
