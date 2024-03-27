import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from astropy.table import Table

HELPERS_DIR = Path(__file__).parent
sys.path.append(str(HELPERS_DIR.parent))  # put code_src dir on the path

# Lazy-load all other imports to avoid depending on modules that will not actually be used.

ARCHIVE_NAMES = {
    "all": ["Gaia", "HCV", "HEASARC", "IceCube", "PanSTARRS", "TESS_Kepler", "WISE", "ZTF"],
    "core": ["Gaia", "HEASARC", "IceCube", "WISE", "ZTF"],
}
DEFAULTS = {
    "run_id": "my-run",
    "get_samples": ["Yang"],
    "consolidate_nearby_objects": True,
    "overwrite_existing_sample": True,
    "archives": ARCHIVE_NAMES["all"],
    "overwrite_existing_lightcurves": True,
    "use_yaml": False,
    "yaml_filename": "kwargs.yml",
    "sample_filename": "object_sample.ecsv",
    "parquet_dataset_name": "lightcurves.parquet",
}


def run(*, build, **kwargs_dict):
    """Run the light_curve_generator step indicated by `build`.

    Parameters
    ==========

    build : str
        Which step to run. Generally either "sample" or "lightcurves". Can also be either "kwargs"
        or any of the `kwargs_dict` keys, and then the full set of kwargs is built from `kwargs_dict`
        and either the whole dictionary is returned (if "kwargs") or the value of this key is returned.

    kwargs_dict
        Key/value pairs for the build function. This can include any key in the dict
        `helpers.scale_up.DEFAULTS` plus "archive". These are described below.

        run_id : str
            ID for this run. This is used to name the output subdirectory ("base_dir") where the
            scale_up helper will read/write files.

        get_samples : list or dict[dict]
            Names of get_<name>_sample functions from which to gather the object sample.
            To send keyword arguments for any of the named functions, use a dict with key=name
             value=dict of keyword arguments for the named function). Defaults will be
            used for any parameter not provided.

        consolidate_nearby_objects : bool
            Whether to consolidate nearby objects in the sample. Passed to the clean_sample function.

        overwrite_existing_sample : bool
            Whether to overwrite an existing .ecsv file. If false and the file exists, the sample will simply
            be loaded from the file and returned.

        archive : str
            Name of a <name>_get_lightcurves archive function to call when building light curves.

        archives : list or dict[dict]
            Names of <name>_get_lightcurves functions. Use a dict (key=name, value=dict of kwargs for the named
            function) to override defaults for the named function. **Note that this does not determine which
            archives are actually called** (unlike `get_samples` with sample names). This is because the
            scale_up helper can only run one function at a time, so a single archive name must be passed separately
            when building light curves.

        overwrite_existing_lightcurves : bool
            Whether to overwrite an existing .parquet file. If false and the file exists, the light curve
            data will simply be loaded from the file and returned.

        use_yaml : bool
            Whether to load additional kwargs from a yaml file.

        yaml_filename : str
            Name of the yaml to read/write, relative to the base_dir.

        sample_filename: str
            Name of the `sample_table` .ecsv file to read/write, relative to the base_dir.

        parquet_dataset_name : str
            Name of the directory to read/write the parquet dataset, relative to the base_dir.
            The dataset will contain one .parquet file for each archive that returned light curve data.
    """
    my_kwargs_dict = _construct_kwargs_dict(**kwargs_dict)

    if build == "sample":
        return _build_sample(**my_kwargs_dict)

    if build == "lightcurves":
        return _build_lightcurves(**my_kwargs_dict)

    return _build_other(keyword=build, **my_kwargs_dict)


# ---- build functions ---- #


def _build_sample(*, get_samples, consolidate_nearby_objects, sample_file, overwrite_existing_sample, **_):
    """Build an AGN sample using coordinates from different papers."""
    _init_worker(job_name="build=sample")

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if sample_file.is_file() and not overwrite_existing_sample:
        print(f"Using existing object sample at: {sample_file}", flush=True)
        return Table.read(sample_file, format="ascii.ecsv")

    # else continue fetching the sample
    print(f"Building object sample from: {list(get_samples.keys())}", flush=True)

    import sample_selection

    # list of tuples. tuple contains: (get-sample function, kwargs dict for that function)
    get_sample_functions = [
        (getattr(sample_selection, f"get_{name}_sample"), kwargs) for name, kwargs in get_samples.items()
    ]

    # iterate over the functions and get the samples
    coords, labels = [], []
    for get_sample_fnc, kwargs in get_sample_functions:
        get_sample_fnc(coords, labels, **kwargs)

    # create an astropy Table of objects
    sample_table = sample_selection.clean_sample(coords, labels, consolidate_nearby_objects=consolidate_nearby_objects)

    # save and return the Table
    sample_table.write(sample_file, format="ascii.ecsv", overwrite=True)
    print(f"Object sample saved to: {sample_file}")
    print(_now(), flush=True)
    return sample_table


def _build_lightcurves(*, archive, archive_kwargs, sample_file, parquet_dir, overwrite_existing_lightcurves, **_):
    """Fetch data from the archive and build light curves for objects in sample_filename."""
    _init_worker(job_name=f"build=lightcurves, archive={archive}")
    parquet_filepath = parquet_dir / f"archive={archive}" / "part0.snappy.parquet"

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if parquet_filepath.is_file() and not overwrite_existing_lightcurves:
        from data_structures import MultiIndexDFObject

        print(f"Using existing light curve data at: {parquet_filepath}", flush=True)
        return MultiIndexDFObject(data=pd.read_parquet(parquet_filepath))

    # Load the sample.
    sample_table = Table.read(sample_file, format="ascii.ecsv")

    # Import only the archive module that will actually be used to avoid unnecessary dependencies.
    archive_functions = importlib.import_module(f"{archive}_functions")
    get_lightcurves_fnc = getattr(archive_functions, f"{archive}_get_lightcurves")
    # archive_kwargs = archive_kwargs.get(f"{archive}_get_lightcurves", {})

    # Query the archive and load light curves.
    lightcurve_df = get_lightcurves_fnc(sample_table, **archive_kwargs)

    # Save and return the light curve data or tell the user there is no data.
    if len(lightcurve_df.data.index) == 0:
        print(f"No light curve data was returned from {archive}.")
        return
    parquet_filepath.parent.mkdir(parents=True, exist_ok=True)
    lightcurve_df.data.to_parquet(parquet_filepath)
    print(f"Light curves saved to:\n\tparquet_dir={parquet_dir}\n\tfile={parquet_filepath.relative_to(parquet_dir)}")
    print(_now(), flush=True)
    return lightcurve_df


def _build_other(keyword, **kwargs_dict):
    if keyword == "kwargs":
        return kwargs_dict

    # if this was called from the command line, we need to print the value so it can be captured by the script
    # this is indicated by a "+" flag appended to the keyword
    print_scalar, print_list = keyword.endswith("+"), keyword.endswith("+l")
    my_keyword = keyword.removesuffix("+l").removesuffix("+")

    # get the keyword value
    if my_keyword in ["archive_names_all", "archive_names_core"]:
        value = ARCHIVE_NAMES[my_keyword.split("_")[-1]]
    else:
        value = kwargs_dict[my_keyword]

    if print_scalar:
        print(value)
    if print_list:
        print(" ".join(value))

    return value


# ---- construct kwargs ---- #


def _construct_kwargs_dict(**kwargs_dict):
    """Construct a complete kwargs dict by combining defaults, yaml (if requested), and `kwargs_dict`
    (listed in order of increasing precedence).
    """
    run_id = kwargs_dict.get("run_id", DEFAULTS["run_id"])
    base_dir = HELPERS_DIR.parent.parent / f"output/lightcurves-{run_id}"

    # load kwargs from yaml if requested
    yaml_file = base_dir / kwargs_dict.get("yaml_filename", DEFAULTS["yaml_filename"])
    my_kwargs = _load_yaml(yaml_file) if kwargs_dict.get("use_yaml", DEFAULTS["use_yaml"]) else {}

    # update with kwargs_dict. both may contain dict values for the following keys, which need deep updates.
    for key in ["get_samples", "archives"]:
        my_kwargs[key] = _deep_update_kwargs_group(key, my_kwargs.pop(key, []), kwargs_dict.pop(key, []))
    my_kwargs.update(kwargs_dict)  # update with any additional keys in kwargs_dict

    # add any defaults that are still missing
    for key in set(DEFAULTS) - set(my_kwargs):
        my_kwargs[key] = DEFAULTS[key]

    # if a single archive is requested, lower-case it and pull out the right set of kwargs for _build_lightcurves
    if my_kwargs.get("archive"):
        my_kwargs["archive"] = my_kwargs["archive"].lower()
        my_kwargs["archive_kwargs"] = my_kwargs["archives"].get(my_kwargs["archive"], {})

    # set path kwargs and make the base dir if needed
    my_kwargs["run_id"] = run_id
    my_kwargs["base_dir"] = base_dir
    my_kwargs["logs_dir"] = base_dir / "logs"
    my_kwargs["sample_file"] = base_dir / my_kwargs["sample_filename"]
    my_kwargs["parquet_dir"] = base_dir / my_kwargs["parquet_dataset_name"]
    my_kwargs["yaml_file"] = yaml_file
    base_dir.mkdir(parents=True, exist_ok=True)

    # sort by key and return
    return {key: my_kwargs[key] for key in sorted(my_kwargs)}


def _deep_update_kwargs_group(key, group_a, group_b):
    # if both groups are empty, just return defaults
    if len(group_a) == 0 and len(group_b) == 0:
        return _kwargs_list_to_dict(DEFAULTS[key])

    # these groups may be either lists or dicts.
    # turn them both into dicts with key = name, value = dict of kwargs for name (empty dict if none supplied)
    my_group_a, my_group_b = _kwargs_list_to_dict(group_a), _kwargs_list_to_dict(group_b)

    # update a with b. first, descend one level to update individual name/kwarg pairs.
    for name in my_group_a:
        my_group_a[name].update(my_group_b.pop(name, {}))
    # add any keys from b that were not in a
    my_group_a.update(my_group_b)

    return my_group_a


def _kwargs_list_to_dict(list_or_dict):
    if isinstance(list_or_dict, list):
        return {name.lower(): {} for name in list_or_dict}
    return {name.lower(): kwargs for name, kwargs in list_or_dict.items()}


# ---- other utils ---- #


def _init_worker(job_name="worker"):
    """Run generic start-up tasks for a job."""
    # print the Process ID for the current worker so it can be killed if needed
    print(f"{_now()} | [pid={os.getpid()}] Starting {job_name}", flush=True)


def _now():
    # parse this datetime using: dateutil.parser.parse(_now())
    date_format = "%Y/%m/%d %H:%M:%S %Z"
    return datetime.now(timezone.utc).strftime(date_format)


def _load_yaml(yaml_file):
    with open(yaml_file, "r") as fin:
        yaml_dict = yaml.safe_load(fin)
    return yaml_dict


def write_kwargs_to_yaml(**kwargs_dict) -> None:
    """Write `kwargs_dict` as a yaml file in the run's base_dir."""
    yaml_path = run(build="yaml_file", **kwargs_dict)
    with open(yaml_path, "w") as fout:
        yaml.safe_dump(kwargs_dict, fout)
    print(f"kwargs written to {yaml_path}")


# ---- helpers for __name__ == "__main__" ---- #


def _argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        type=str,
        default="sample",
        help="Either 'sample', 'lightcurves', or a kwargs key.",
    )
    parser.add_argument(
        "--kwargs_dict",
        type=str,
        default=list(),
        nargs="*",
        help="Keyword arguments for the run. Input as a list of strings 'key=value'.",
    )
    parser.add_argument(
        "--kwargs_json",
        type=json.loads,
        default=r"{}",
        help="Kwargs as a json string, to be added to kwargs_dict.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Archive name to query for light curves, to be added to kwargs_dict.",
    )
    return parser


def _parse_args(args_list):
    args = _argparser().parse_args(args_list)

    # start with kwargs_json, then update
    my_kwargs_dict = args.kwargs_json

    # parse args.kwargs_dict 'key=value' pairs into a proper dict and convert true/false to bool
    bool_map = {"true": True, "false": False}
    args_kwargs_dict = {
        key: bool_map.get(val.lower(), val)
        for (key, val) in dict(kwarg.split("=") for kwarg in args.kwargs_dict).items()
        # for (key, val) in {kwarg.split("=")[0]: kwarg.split("=")[1] for kwarg in args.kwargs_dict}.items()
    }

    # update my_kwargs_dict with args_kwargs_dict
    # both may contain dict values for the following keys, which need deep updates.
    for key in list(k for k in ["get_samples", "archives"] if k in my_kwargs_dict):
        my_kwargs_dict[key] = _deep_update_kwargs_group(key, my_kwargs_dict.pop(key), args_kwargs_dict.pop(key, []))
    my_kwargs_dict.update(args_kwargs_dict)  # update with any additional keys in args_kwargs_dict

    # add the archive, if provided
    if args.archive:
        my_kwargs_dict["archive"] = args.archive

    return args.build, my_kwargs_dict


if __name__ == "__main__":
    build, kwargs_dict = _parse_args(sys.argv[1:])
    run(build=build, **kwargs_dict)
