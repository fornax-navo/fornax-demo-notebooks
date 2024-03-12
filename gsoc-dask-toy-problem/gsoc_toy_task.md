---
jupytext:
  formats: ipynb,md:myst
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

# Install all the necessary packages

```{code-cell} ipython3
# !pip install numpy pandas datetime matplotlib dask distributed #{if you want to install on whole system}
# !conda install numpy pandas dask matplotlib distributed -y #{if you want to install on the specific conda environment}
# I chose conda{except for datetime, as it wasn't available at anaconda}, so as to prevent any dependency issues, and to keep all packages organized at a specific place
```

# Make the necessary imports for running the functions

```{code-cell} ipython3
import archives  # Importing the pre-built functions from the repository
import pandas as pd  # Importing pandas for data manipulation
import numpy as np 
import datetime  # Importing datetime for time-related operations
import warnings # Importing this module to filter out warnings related to using new ports for each dask process
import matplotlib.pyplot as plt # Importing this so as to draw graphs for comparison
from dask.distributed import Client, LocalCluster # Importing this for running dask process on remote server
warnings.filterwarnings("ignore", category = UserWarning)
```

# Defining the functions: serial & dask

```{code-cell} ipython3
# Function to retrieve lightcurves serially
def get_lightcurves_serial(num_samples: int) -> datetime.datetime:
    # Record the starting time
    start_time = datetime.datetime.now()

    # Retrieving lightcurves from various sources
    gaia_lightcurves = archives.get_gaia_lightcurves(num_samples)
    heasarc_lightcurves = archives.get_heasarc_lightcurves(num_samples)
    wise_lightcurves = archives.get_wise_lightcurves(num_samples)
    ztf_lightcurves = archives.get_ztf_lightcurves(num_samples)
    
    # Concatenating all the retrieved lightcurves into a single DataFrame
    all_lightcurves_df = pd.concat([gaia_lightcurves, heasarc_lightcurves, wise_lightcurves, ztf_lightcurves])
    
    # Calculating the time taken for the operation
    elapsed_time = datetime.datetime.now() - start_time
    
    # Returning the elapsed time
    return elapsed_time
```

```{code-cell} ipython3
def get_lightcurves_dask(num_samples: int) -> datetime.timedelta:
    # Start Dask cluster with a specific port and avoid daemonic processes
    dask_cluster = LocalCluster(processes=False)
    dask_client = Client(dask_cluster)

    # Define the function to parallelize
    def get_lightcurves_parallel(num_samples, func):
        return func(num_samples)

    # Record the starting time
    start_time = datetime.datetime.now()

    # Parallelize the functions using Dask
    gaia_future = dask_client.submit(get_lightcurves_parallel, num_samples, archives.get_gaia_lightcurves)
    heasarc_future = dask_client.submit(get_lightcurves_parallel, num_samples, archives.get_heasarc_lightcurves)
    wise_future = dask_client.submit(get_lightcurves_parallel, num_samples, archives.get_wise_lightcurves)
    ztf_future = dask_client.submit(get_lightcurves_parallel, num_samples, archives.get_ztf_lightcurves)

    # Gather results
    gaia_df = gaia_future.result()
    heasarc_df = heasarc_future.result()
    wise_df = wise_future.result()
    ztf_df = ztf_future.result()

    # Concatenate the results into a single Pandas DataFrame
    lightcurves_df = pd.concat([gaia_df, heasarc_df, wise_df, ztf_df])

    # Calculate the elapsed time
    elapsed_time = datetime.datetime.now() - start_time

    # Shut down the Dask cluster
    dask_client.shutdown()

    # Return the elapsed time
    return elapsed_time
```

# Required Task: Starting a Dask cluster and stopping it when finished.

```{code-cell} ipython3
# To test locally the working of serial as well as dask pipelines:

# Assume sample_size is 5
sample_size = 5

#Serial Function
time_elapsed_in_serial = get_lightcurves_serial(sample_size).total_seconds()
print(f"Time elapsed in serial pipeline: {time_elapsed_in_serial}")

#Dask Function
time_elapsed_in_dask = get_lightcurves_dask(sample_size).total_seconds()
print(f"Time elapsed in dask pipeline: {time_elapsed_in_dask}")
```

# Optional Task: Executing the four archives functions and comparing the processing times for Dask and Serial pipelines

+++

# Running the functions and appending the processing times to a list

```{code-cell} ipython3
# Define data
num_of_samples = [5, 10, 15, 20]  # Sample sizes

# Initiate an empty list for processing times
time_for_lightcurves_in_serial = []
time_for_lightcurves_in_dask = []

# Loop over all sample sizes to get the processing time and append it to respective list
for sample_size in num_of_samples:
    print(f"Processing for sample size: {sample_size}")
    
    #Serial Function
    time_elapsed_in_serial = get_lightcurves_serial(sample_size).total_seconds()
    time_for_lightcurves_in_serial.append(time_elapsed_in_serial)
    print(f"Serial Completed for sample size: {sample_size}")
    
    #Dask Function
    time_elapsed_in_dask = get_lightcurves_dask(sample_size).total_seconds()
    time_for_lightcurves_in_dask.append(time_elapsed_in_dask)
    print(f"Dask Completed for sample size: {sample_size}")
```

# Plotting the processing time from the list on a Graph for visualization

```{code-cell} ipython3
# Plot
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(num_of_samples, time_for_lightcurves_in_serial, 's-', color='#FF6347', label='Serial', linewidth=2, markersize=8)
plt.plot(num_of_samples, time_for_lightcurves_in_dask, 'o-', color='#4169E1', label='Dask', linewidth=2, markersize=8)

# Add legend
plt.legend(fontsize=12)

# Add labels and title
plt.xlabel('Number of Samples', fontsize=14)  # X-axis label
plt.ylabel('Time (s)', fontsize=14)  # Y-axis label
plt.xticks(fontsize=12)  # X-axis ticks font size
plt.yticks(fontsize=12)  # Y-axis ticks font size
plt.title('Time to Retrieve Lightcurves', fontsize=16)  # Plot title

# Add grid
plt.grid(True, linestyle='--', alpha=0.6)

# Add annotations
for i, (serial_time, dask_time) in enumerate(zip(time_for_lightcurves_in_serial, time_for_lightcurves_in_dask)):
    plt.text(num_of_samples[i], serial_time, f'{serial_time} s', ha='right', va='bottom', fontsize=10, color='#FF6347')
    plt.text(num_of_samples[i], dask_time, f'{dask_time} s', ha='right', va='bottom', fontsize=10, color='#4169E1')

# Show plot
plt.tight_layout()  # Adjust layout
plt.show()
```
