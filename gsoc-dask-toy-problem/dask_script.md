---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

### Serial execution

```{code-cell} ipython3
import archives  # be sure to import this **first**
import pandas as pd
import datetime

num_sample = 5  # may vary between about 5 and 500,000
def get_lc_serial(num_sample: int)->datetime.datetime:
    to = datetime.datetime.now()
    gaia_df = archives.get_gaia_lightcurves(num_sample)
    heasarc_df = archives.get_heasarc_lightcurves(num_sample)
    wise_df = archives.get_wise_lightcurves(num_sample)
    ztf_df = archives.get_ztf_lightcurves(num_sample)
    lightcurves_df = pd.concat([gaia_df, heasarc_df, wise_df, ztf_df])
    return datetime.datetime.now() - to
```

### Dask parallel execution

```{code-cell} ipython3
from dask.distributed import Client, LocalCluster
import pandas as pd
import archives
import datetime as dt

def get_lc_dask(num_sample: int)->datetime.datetime:
    # Start Dask cluster with a specific port and avoid daemonic processes
    cluster = LocalCluster(processes=False)
    client = Client(cluster)

    # Define the function to parallelize
    def get_lightcurves(num_sample, func):
        return func(num_sample)

    # Define the sample size
    num_sample = 5
    to = dt.datetime.now()
    # Parallelize the functions using Dask
    gaia_future = client.submit(get_lightcurves, num_sample, archives.get_gaia_lightcurves)
    heasarc_future = client.submit(get_lightcurves, num_sample, archives.get_heasarc_lightcurves)
    wise_future = client.submit(get_lightcurves, num_sample, archives.get_wise_lightcurves)
    ztf_future = client.submit(get_lightcurves, num_sample, archives.get_ztf_lightcurves)

    # Gather results
    gaia_df = gaia_future.result()
    heasarc_df = heasarc_future.result()
    wise_df = wise_future.result()
    ztf_df = ztf_future.result()

    # Concatenate the results into a single Pandas DataFrame
    lightcurves_df = pd.concat([gaia_df, heasarc_df, wise_df, ztf_df])
    return dt.datetime.now()-to
```

```{code-cell} ipython3
import numpy as np

x =  [5,10]
y_lc_serial = []
y_lc_dask = []
for each in x:
    y_lc_dask.append(get_lc_dask(int(each)).total_seconds())
    y_lc_serial.append(get_lc_serial(int(each)).total_seconds())
```

```{code-cell} ipython3

import matplotlib.pyplot as plt
plt.plot(x, y_lc_serial, 'ro-', label='Serial')
plt.plot(x, y_lc_dask, 'bo-', label='Dask')
plt.legend()
plt.xlabel('Number of samples')
plt.ylabel('Time (s)')
plt.xticks(x)
plt.title('Time to get lightcurves')
plt.show()
```
