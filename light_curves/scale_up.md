---
jupytext:
  cell_metadata_filter: -all
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

# Make Multi-Wavelength Light Curves for Large Samples

+++

## Learning Goals

By the end of this tutorial, you will be able to:

- Parallelize the code demonstrated in the [light_curve_generator](light_curve_generator.md) notebook to get multi-wavelength light curves.
- Launch a run using a large sample of objects, monitor the run's progress automatically, and understand its resource usage (CPU and RAM).
- Understand some general challenges and requirements when scaling up code.

+++

## Introduction

+++

This notebook shows how to collect multi-wavelength light curves for a large sample of target objects.
This is a scaling-up of the [light_curve_generator](light_curve_generator.md) and assumes you are familiar with the content of that notebook.
This code should be used for sample sizes greater than a few hundred.
Many of the challenges, needs, and wants discussed in the appendix and addressed throughout the notebook are common when scaling up code and the ideas can be applied to other use cases.

Notebook sections are:

- "Overview" describes functionality of the included bash script and python helper functions. Compares some parallel processing options.
- "Example 1" shows how to launch a large-scale run using the bash script, monitor its progress automatically, and diagnose a problem (out of RAM).
- "Example 2" shows how to parallelize the example from the light_curve_generator notebook using the helper and python's `multiprocessing` library.
- "Example 3" details the helper parameter options and how to use them in python and bash.
- "Appendix" contains background information including a discussion of the challenges, needs, and wants encountered when scaling up this code, and general advice for the user.

**Many of the bash commands below are shown in non-executable cells because they are not intended to be run in this notebook.**
Bash commands that are not executed below begin with the symbol `$ `, and those that are executed begin with `!`.
Both types can be called from the command-line -- open a new terminal and copy/paste the cell text without the beginning symbol. The bash script is not intended to be executed from within a notebook and may behave strangely if attempted.
Also be aware that the script path shown in the commands below assumes you are in the same directory as this notebook. Adjust it if needed.

+++

## Overview

+++

### Parallel processing methods: bash script vs. python's `multiprocessing`

+++

- Bash script. Recommended for most runs with medium to large samples. Allows ZTF to use additional parallelization internally, and so is often faster (ZTF often takes the longest and returns the most data for AGN-like samples). Writes stdout and stderr to log files, useful for monitoring jobs and resource usage. Can save `top` output to a file to help identify CPU and RAM usage/needs.
- Python's `multiprocessing` library. May be convenient for runs with small to medium sample sizes (< few thousand). Has drawbacks that may be significant including the inability to use ZTF's internal parallelization and that it does not save the log output (stdout and stderr) to file. An advantage of the `multiprocessing` example in this notebook over the light_curve_generator is that it automatically saves the sample and light curve data to disk after loading them and can automatically skip those functions in subsequent calls and use the files instead.

### The python helper

+++

The python "helper" is a set of wrapper functions around the same 'code_src/' functions used in the light_curve_generator notebook.

- The wrappers facilitate parallelization and large-scale runs by automating tasks like saving the function outputs to files.
- The helper does not actually implement parallelization and can only run one function per call.
- The helper can be used in combination with any parallel processing method.
- The helper can load `top` output from a file to pandas DataFrames and make some figures.

### The bash script

The bash script allows the user to collect light curves from a large scale sample with a single command and provides options to help manage and monitor the run.
In a nutshell, the script does the following when called using flags that "launch a run":

- calls the helper to gather the requested sample and then launches jobs for each archive query in separate, parallel processes.
- redirects stdout and stderr to log files.
- tells the user what the process PIDs are and where the log and data files are.
- exits, leaving the archive jobs running in the background.

In case all the archive calls need to be canceled after the script exits, the user can call the script again with a different flag to have it find and kill all the processes it launched.

While the jobs are running, the user can call the script again with a different flag to have it save `top` output to a log file at a user-defined interval.
The helper can be used to load this file to pandas DataFrames and make some figures.

+++

### Imports

```{code-cell}
# Ensure all dependencies are installed
!pip install -r requirements.txt
```

```{code-cell}
import multiprocessing  # python parallelization method
import pandas as pd  # use a DataFrame to work with light-curve and other data
import sys  # add code directories to the path

sys.path.append("code_src/")
import helpers.scale_up  # python "helper" for parallelization and large scale runs
import helpers.top  # load `top` output to DataFrames and make figures
from data_structures import MultiIndexDFObject  # load light curve data as a MultiIndexDFObject
from plot_functions import create_figures  # make light curve figures
```

## Example 1: Multi-wavelength light curves for 500,000 SDSS AGN

+++

This example shows how to launch a large-scale collection of light curves using the bash script, monitor its performance, and diagnose a problem (out of RAM).
This run collects light curves for 500,000 SDSS objects and takes several hours to complete, but is not actually executed here.
Instead, we show the bash commands and then look at logs that were generated by running the commands on 2024/03/01.

+++

### Launch the run

+++

```bash
# This will launch a very large scale run.
# See Example 3 for a detailed explanation of parameter options.
# If you are running this for the first time, reduce the sample size 500000 -> 50.

# run_id will be used in future calls to manage this run, and also determines the name of the output directory.
$ run_id="demo-SDSS-500k"

# Execute the run.
# This will get the sample and then call all archives specified by the '-a' flag.
# Defaults will be used for all keyword arguments that are not specified.
$ bash code_src/helpers/scale_up.sh \
    -r "$run_id" \
    -j '{"get_samples": {"SDSS": {"num": 500000}}, "archives": {"ZTF": {"nworkers": 8}}}' \
    -a "core"  # shortcut for "Gaia HEASARC IceCube WISE ZTF"
```

+++

The script will run the 'get sample' job, then launch the archive query jobs in parallel and exit.
Archive jobs will continue running in the background until they either complete or encounter an error.
Example 2 shows how to load the data.

Command output from 2024/03/01 logs:

```{code-cell}
!cat output/lightcurves-demo-SDSS-500k/logs/scale_up.sh.log
```

### Cancel

+++

You can cancel jobs at any time.

If the script is still running, press `Control-C`.

If the script has exited, there are two options.

1) To cancel an individual job, kill the job's process using:

+++

```bash
$ pid=0000  # get the number from script output
$ kill $pid
```

+++

2) To cancel the entire run, use the `-k` (kill) flag:

+++

```bash
$ bash code_src/helpers/scale_up.sh -r "$run_id" -k
```

+++

### Restart

+++

If you want to restart and skip step(s) that previously completed, run the first command again and add one or both "overwrite" flags set to false:

+++

```bash
# use the same run_id as before
$ bash code_src/helpers/scale_up.sh \
    -r "$run_id" \
    -j '{"get_samples": {"SDSS": {"num": 500000}}, "archives": {"ZTF": {"nworkers": 8}}}' \
    -a "core" \
    -d "overwrite_existing_sample=false" \
    -d "overwrite_existing_lightcurves=false"
```

+++

### Monitor

+++

There are at least three places to look for information about a run's status.

- Check the logs for job status or errors. The bash script will redirect stdout and stderr to log files and print out the paths for you.
- Check for light curve (parquet) data. The script will print out the "parquet_dir". `ls` this directory. You will see a subdirectory for each archive call that has completed successfully, assuming it retrieved data for the sample.
-  Watch `top`. The script will print the job PIDs. The script can also monitor `top` for you and save the output to a log file.

+++

#### Logs

```{code-cell}
# Gaia log from 2024/03/01 (success)
!cat output/lightcurves-demo-SDSS-500k/logs/gaia.log
```

```{code-cell}
# ZTF log from 2024/03/01 (failure)
!cat output/lightcurves-demo-SDSS-500k/logs/ztf.log
```

The logs above show that the ZTF job loaded the light curve data successfully ("100%") but exited without writing the parquet file (no "Light curves saved" message like Gaia).
The data was lost.
There is also no indication of an error; the job just ends.
We can diagnose what happened by looking at the `top` output.

+++

#### `top`

While the jobs are running, you can monitor `top` and save the output to a log file by calling the script again with the `-t` flag.
This call must be separate from the call that launches the run.
If you want to capture the sample step as well as the archive queries, open a second terminal and call the script with `-t` right after launching the run in the first terminal.

+++

```bash
$ interval=10m  # choose your interval
$ bash code_src/helpers/scale_up.sh -r "run_id" -t "$interval"
```

+++

The script will continue running until after all of run_id's jobs have completed.
You can cancel at anytime with `Control-C` and start it again with a new interval.

Once saved to file, the helper can parse the `top` output into pandas DataFrames and make some figures.

`top` output from 2024/03/01:

```{code-cell}
run_id = "demo-SDSS-500k"
logs_dir = helpers.scale_up.run(build="logs_dir", run_id=run_id)

toplog = helpers.top.load_top_output(toptxt_dir=logs_dir, run_id=run_id)
```

```{code-cell}
toplog.summary_df.sample(5)
```

```{code-cell}
toplog.pids_df.sample(5)
```

```{code-cell}
fig = toplog.plot_overview()
```

In the figure above, "Total" is the machine total, the "%CPU" panel is grouped by PID (process ID), and the "%MEM" panel is grouped by job.
There are many interesting features in the figure that the reader may want to look at in more detail.
For example, other than ZTF's memory spike at the end, we see that the full run collecting multi-wavelength light curves for the SDSS 500k sample could be completed with about 2 CPU and 60G RAM.

We want to learn why the ZTF job failed.
Let's zoom in on that time period:

```{code-cell}
fig = toplog.plot_overview(between_time=("22:55", "23:10"))
```

In the second panel above we see the ZTF worker processes (which load the light curve data) ending just after 23:00 and then the ZTF parent process continues by itself.
Around 23:06 in the fourth panel, we see the ZTF job's memory usage rise sharply to almost 100% and then drop immediately to zero when the job terminates.
This coincides with the total available memory dropping to near zero in the third panel.
This shows that the machine did not have enough memory for the ZTF call to successfully transform the light curve data collected by the workers into a `MultiIndexDFObject` and write it as a parquet file, so the machine killed the job.

The solution is to rerun the ZTF archive call on a machine with more RAM.

To learn exactly which step in `ztf_get_lightcurves` was causing this and how much memory it actually needed, additional print statements were inserted into the code similar to the following:

```{code-cell}
print(f"{helpers.scale_up._now()} | starting explode", flush=True)  # _now() prints the current timestamp
```

ZTF was then rerun on a large machine and `top` output was saved.
After the run, we manually compared timestamps between the ZTF log and `top` output and tagged relevant `top` timestamps with corresponding step names by appending the name to the '----' delineator, like this for the "explode" step:

```{code-cell}
!cat output/lightcurves-demo-SDSS-500k/logs/top.tag-ztf.txt | grep -A12 explode
```

The helper can recognize these tags and show them on a figure:

```{code-cell}
ztf_toplog = helpers.top.load_top_output(toptxt_file="top.tag-ztf.txt", toptxt_dir=logs_dir, run_id=run_id)

fig = ztf_toplog.plot_time_tags(summary_y="used_GiB")
# (This run starts by reading a previously cached parquet file containing the raw data returned by workers.)
```

This figure shows that almost 100G RAM is required for the ZTF job to succeed.
It further shows that the "explode" step requires the most memory, followed by creating the MultiIndexDFObject.
From here, the user can choose an appropriately sized machine and/or consider whether `ztf_get_lightcurves` could be made to use less memory.

+++

## Example 2: Parallelizing the light_curve_generator notebook

+++

This example shows how to parallelize the example from the light_curve_generator notebook using the helper and python's `multiprocessing`.
An advantage of the example shown here is that it automatically saves the sample and light curve data to disk after loading them and can automatically skip those steps in subsequent calls and use the files instead.

Define the keyword arguments for the run:

```{code-cell}
kwargs_dict = {
    "run_id": "demo-Yang-sample",
    # Paper names to gather the sample from.
    "get_samples": ["Yang"],
    # Keyword arguments for *_get_lightcurves archive calls.
    "archives": {
        "Gaia": {"search_radius": 1 / 3600, "verbose": 0},
        "HEASARC": {"catalog_error_radii": {"FERMIGTRIG": 1.0, "SAXGRBMGRB": 3.0}},
        "IceCube": {"icecube_select_topN": 3, "max_search_radius": 2.0},
        "WISE": {"radius": 1.0, "bandlist": ["W1", "W2"]},
        "ZTF": {"match_radius": 1 / 3600, "nworkers": None},
    },
}
# See Example 3 for a detailed explanation of parameter options.
kwargs_dict
```

Decide which archives to query.
This is a separate list because the helper can only run one archive call at a time.
We will iterate over this list and launch each job separately.

```{code-cell}
# archive_names = ["PanSTARRS", "WISE"]  # choose your own list
archive_names = helpers.scale_up.ARCHIVE_NAMES["all"]  # predefined list ("core" or "all")
archive_names
```

Collect the sample and write it as a .ecsv file.
Then query the archives in parallel using a `multiprocessing.Pool` and write the light curve data as .parquet files.

```{code-cell}
%%time
sample_table = helpers.scale_up.run(build="sample", **kwargs_dict)
# sample_table is returned if you want to look at it but it is not used below

with multiprocessing.Pool(processes=len(archive_names)) as pool:
    # submit one job per archive
    for archive in archive_names:
        pool.apply_async(helpers.scale_up.run, kwds={"build": "lightcurves", "archive": archive, **kwargs_dict})
    pool.close()  # signal that no more jobs will be submitted to the pool
    pool.join()  # wait for all jobs to complete

# Note: The console output from different archive calls gets jumbled together below.
# Worse, error messages tend to get lost in the background and never displayed.
# If you have trouble, consider running an archive call individually without the Pool
# or using the bash script instead.
```

The light curve data is saved as a parquet dataset in the "parquet_dir" directory.
Load it:

```{code-cell}
# copy/paste the directory path from the output above, or ask the helper for it like this:
parquet_dir = helpers.scale_up.run(build="parquet_dir", **kwargs_dict)
df_lc = pd.read_parquet(parquet_dir)

df_lc.sample(10)
```

Now we can make figures:

```{code-cell}
_ = create_figures(df_lc=MultiIndexDFObject(data=df_lc), show_nbr_figures=1, save_output=False)
```

## Example 3: Keyword arguments and script flags

+++

This example shows the python `kwargs_dict` and bash script flag options in more detail.

### Python `kwargs_dict`

`kwargs_dict` is a dictionary containing all keyword arguments for the run. It can contain:

- names and keyword arguments for any of the `get_*_sample` functions.
- keyword arguments for any of the `*_get_lightcurves` functions.
- other keyword arguments used directly by the helper.
  These options and their defaults are shown below, further documented in the helper's `run` function.

```{code-cell}
# show kwargs_dict defaults
helpers.scale_up.DEFAULTS
```

```{code-cell}
# show parameter documentation
print(helpers.scale_up.run.__doc__)
```

### Bash script flags

Use the `-h` (help) flag to view the script's flag options:

```{code-cell}
# show flag documentation
!bash code_src/helpers/scale_up.sh -h
```

### Using a yaml file

It can be convenient to save the parameters in a yaml file, especially when using the bash script or in cases where you want to store parameters for later reference or re-use.

Define an extended set of parameters and save it as a yaml file:

```{code-cell}
yaml_run_id = "demo-kwargs-yaml"

get_samples = {
    "green": {},
    "ruan": {},
    "papers_list": {
        "paper_kwargs": [
            {"paper_link": "2022ApJ...933...37W", "label": "Galex variable 22"},
            {"paper_link": "2020ApJ...896...10B", "label": "Palomar variable 20"},
        ]
    },
    "SDSS": {"num": 10, "zmin": 0.5, "zmax": 2, "randomize_z": True},
    "ZTF_objectid": {"objectids": ["ZTF18aabtxvd", "ZTF18aahqkbt", "ZTF18abxftqm", "ZTF18acaqdaa"]},
}

archives = {
    "Gaia": {"search_radius": 2 / 3600},
    "HEASARC": {"catalog_error_radii": {"FERMIGTRIG": 1.0, "SAXGRBMGRB": 3.0}},
    "IceCube": {"icecube_select_topN": 4, "max_search_radius": 2.0},
    "WISE": {"radius": 1.5, "bandlist": ["W1", "W2"]},
    "ZTF": {"nworkers": 6, "match_radius": 2 / 3600},
}

yaml_kwargs_dict = {
    "get_samples": get_samples,
    "consolidate_nearby_objects": False,
    "archives": archives,
}

helpers.scale_up.write_kwargs_to_yaml(run_id=yaml_run_id, **yaml_kwargs_dict)
```

The path to the yaml file is printed in the output above.
You can alter the contents of the file as you like.
To use the file, set the kwarg `use_yaml=True`.

Python example for the get-sample step:

```{code-cell}
sample_table = helpers.scale_up.run(build="sample", run_id=yaml_run_id, use_yaml=True)
```

Bash example:

+++

```bash
$ yaml_run_id=demo-kwargs-yaml
$ bash code_src/helpers/scale_up.sh -r "$yaml_run_id" -d "use_yaml=true" -a "core"
```

+++

## Appendix: What to expect

+++

### Challenges of large-scale runs

+++

Scaling up to large sample sizes brings new challenges.
Even if things go smoothly, each function call may need to use a lot more resources like CPU, RAM, and bandwidth, and may take much longer to complete.
Also, we'll want to run some functions in parallel to save time overall, but that will mean they must compete with each other for resources.

These issues are complicated by the fact that different combinations of samples and archive calls can trigger different problems.
Inefficiencies in any part of the process -- our code, archive backends, etc. -- which may have been negligible at small scale can balloon into significant hurdles.

Problems can manifest in different ways.
For example, progress may to slow to a crawl, or it may run smoothly for several hours and then crash suddenly.
If the job is running in the background, print statements and error messages may get lost and never be displayed for the user if they are not redirected to a file.

+++

### Needs and wants for large-scale runs

+++

The main goal is to speed up the run, so we want to look for opportunities to parallelize.
We can group the light_curve_generator code into two main steps: (1) gather the target object sample; then (2) generate light curves by querying the archives and standardizing the returned data.
All of the archive calls have to wait for the sample to be available before starting, but then they can run independently in parallel.

We need to be able to monitor the run's resource usage and capture print statements, error messages, etc. to log files in order to understand if/when something goes wrong.
Even with parallelization, gathering light curves for a large sample of objects is likely to take a few hours at least.
So we want to automate the monitoring tasks as much as possible.

If the run fails, we'd like to be able to restart it without having to redo steps that were previously successful.
To accomplish this, the inputs and outputs need to be less tightly coupled than they are in the light_curve_generator notebook.
Specifically, we want to save the sample and light curve data to file as soon as each piece is collected, and we want the archive functions to be able to get the `sample_table` input from file.

The python helper and bash script were specifically designed to fulfill many of these wants and needs.

+++

### What the user should do

+++

Getting started:
1. Skim this notebook to understand the process, available options, and potential sticking points.
2. Try things with a small sample size first, then scale up to your desired full sample.
3. Don't be surprised if something goes wrong.
    Every new combination of factors can present different challenges and reasons for the code to fail.
    This includes the sample selection, which archives are called and what parameters are used, runtime environment, machine CPU and RAM capabilities, network bandwidth, etc.
    Scaling up any code base comes with challenges, and some of these cannot be fully managed by pre-written code.
    You may need to observe how the code performs, diagnose a problem, and adapt the input parameters, machine size, etc. in order to successfully execute.

To execute a run:
1. Define all of the get-sample and get-lightcurve parameters.
2. Launch the run by calling the bash script or some other multi-processing method.
    Capturing stdout, stderr and `top` output to log files is highly recommended.
4. If a get-lightcurve (get-sample) job exits without writing a .parquet (.ecsv) file, inspect the logs and `top` output to try to determine the reason it failed.
    It could be anything from a missing python library (install it), to an archive encountering an internal error (wait a bit and try again), to the job getting killed prematurely because its needs exceeded the available RAM (try a machine with more RAM, a smaller sample size, or running fewer archive calls at a time), to many other things.

+++

## About this notebook

**Authors**: Troy Raen, Jessica Krick, Brigitta Sipőcz, Shoubaneh Hemmati, Andreas Faisst, David Shupe

**Updated On**: 2024-03-19
