import multiprocessing as mp
import re

import astropy.units as u
import pandas as pd
import pyarrow.fs
import pyarrow.parquet
import pyvo
import tqdm

import helpers.scale_up
from data_structures import MultiIndexDFObject


# the catalog is stored in an AWS S3 bucket
DATARELEASE = "dr18"
BUCKET = "irsa-mast-tike-spitzer-data"
CATALOG_ROOT = f"{BUCKET}/data/ZTF/lc/lc_{DATARELEASE}/"

# get a list of files in the dataset using the checksums file
CATALOG_FILES = (
    pd.read_table(f"s3://{CATALOG_ROOT}checksum.md5", sep="\s+", names=["md5", "path"], usecols=["path"])
    .squeeze()  # there's only 1 column. squeeze it into a Series
    .str.removeprefix("./")
    .to_list()
)


def ztf_get_lightcurves(sample_table, *, nworkers=6, match_radius=1/3600):
    """Function to add the ZTF lightcurves in all three bands to a multiframe data structure.  This is the MAIN function.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    nworkers : int or None
        number of workers in the multiprocessing pool used in the load_lightcurves function.
        This must be None if this function is being called from within a child process already.
        (This function does not support nested multiprocessing.)
    match_radius : float
        search radius (degrees), how far from the source should the archives return results

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """
    # the catalog is in parquet format with one file per ZTF filter, field, ccd, and quadrant
    # use a TAP query to locate which files each object is in
    locations_df = locate_objects(sample_table, match_radius)

    # the catalog is stored in an AWS S3 bucket. loop over the files and load the light curves
    ztf_df = load_lightcurves(locations_df, nworkers=nworkers)

    # if none of the objects were found, the transform_lightcurves function will raise a ValueError
    # so return an empty dataframe now instead of proceeding
    if len(ztf_df.index) == 0:
        return MultiIndexDFObject()

    # clean and transform the data into the form needed for a MultiIndexDFObject
    ztf_df = transform_lightcurves(ztf_df)

    # return the light curves as a MultiIndexDFObject
    indexes, columns = ["objectid", "label", "band", "time"], ["flux", "err"]
    return MultiIndexDFObject(data=ztf_df.set_index(indexes)[columns].sort_index())


def file_name(filtercode, field, ccdid, qid, basedir=None):
    """Lookup the filename for this filtercode, field, ccdid, qid.

    File name syntax starts with: {basedir}/field{field}/ztf_{field}_{filtercode}_c{ccdid}_q{qid}

    Parameters
    ----------
    filtercode : str
        ZTF band name
    field : int
        ZTF field
    ccdid : int
        ZTF CCD
    qid : int
        ZTF quadrant
    basedir : int, optional
        Either 0 or 1. The base directory this file is located in.

    Returns
    -------
    file_name : str
        Parquet file name containing this filtercode, field, ccdid, and qid.

    Raises
    ------
    AssertionError
        if exactly one matching file name is not found in the CATALOG_FILES list
    """
    # if this comes from a TAP query we won't know the basedir,
    # so do a regex search through the CATALOG_FILES list instead
    if basedir is None:
        fre = re.compile(f"[01]/field{field:06}/ztf_{field:06}_{filtercode}_c{ccdid:02}_q{qid}")
        files = [CATALOG_ROOT + f for f in filter(fre.match, CATALOG_FILES)]
        # expecting exactly 1 filename. make it fail if there's more or less.
        assert len(files) == 1, f"found {len(files)} files. expected 1."
        return files[0]

    f = f"{basedir}/field{field:06}/ztf_{field:06}_{filtercode}_c{ccdid:02}_q{qid}_{DATARELEASE}.parquet"
    return CATALOG_ROOT + f


def locate_objects(sample_table, match_radius, chunksize=10000):
    """The catalog's parquet files are organized by filter, field, CCD, and quadrant. Use TAP to look them up.

    https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    match_radius : float
        search radius (degrees), how far from the source should the archives return results
    chunksize : int
        This tap query is much faster when submitting less than ~10,000 coords at a time
        so iterate over chunks of coords_tbl and then concat results.

    Returns
    -------
    locations_df : pd.DataFrame
        Dataframe with ZTF field, CCD, quadrant and other information that identifies each `sample_table`
        object and which parquet files it is in. One row per ZTF objectid.
    """
    # setup for tap query
    tap_service = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    # construct table to be uploaded
    upload_table = sample_table["objectid", "label"]
    upload_table.convert_unicode_to_bytestring()  # TAP requires strings to be encoded
    upload_table["ra"] = sample_table["coord"].ra.deg
    upload_table["dec"] = sample_table["coord"].dec.deg
    # construct the query
    sample_cols = [f"sample.{c}" for c in ["objectid", "label"]]
    ztf_cols = [f"ztf.{c}" for c in ["oid", "filtercode", "field", "ccdid", "qid", "ra", "dec"]]
    select_cols = ', '.join(sample_cols + ztf_cols)
    query = f"""SELECT {select_cols}
        FROM ztf_objects_{DATARELEASE} ztf, TAP_UPLOAD.sample sample
        WHERE CONTAINS(
            POINT('ICRS', sample.ra, sample.dec), CIRCLE('ICRS', ztf.ra, ztf.dec, {match_radius})
        )=1"""

    # do the tap calls
    locations = []
    for i in tqdm.trange(0, len(upload_table), chunksize):
        result = tap_service.run_async(query, uploads={"sample": upload_table[i : i + chunksize]})
        locations.append(result.to_table().to_pandas())

    # locations may contain more than one ZTF object id per band (e.g., yang sample sample_table[11])
    # Sánchez-Sáez et al., 2021 (2021AJ....162..206S)
    # return all the data -- transform_lightcurves will choose which to keep
    return pd.concat(locations, ignore_index=True)


def load_lightcurves(locations_df, nworkers=6, chunksize=100):
    """Loop over the catalog's parquet files (stored in an AWS S3 bucket) and load light curves.

    Parameters
    ----------
    locations_df : pd.DataFrame
        Dataframe with ZTF field, CCD, quadrant and other information that identifies each `coord` object
        and which parquet files it is in.
    nworkers : int or None
        Number of workers in the multiprocessing pool. Use None to turn off multiprocessing. This must be None
        if this function is being called from within a child process (no nested multiprocessing).
    chunksize : int
        Number of files sent to the workers.

    Returns
    -------
    ztf_df : pd.DataFrame
        Dataframe of light curves. Expect one row per oid in locations_df. Each row
        stores a full light curve. Elements in the columns "mag", "hmjd", etc. are arrays.
    """
    # We need to return an empty dataframe if no matches are found. If the TAP query returned matches 
    # but none of them are found in the parquet files, this function will naturally return an empty dataframe.
    # But if the TAP query found no matches, pd.concat (below) will throw a ValueError. Return now to avoid this.
    if len(locations_df.index) == 0:
        return pd.DataFrame()

    # one group per parquet file
    location_grps = locations_df.groupby(["filtercode", "field", "ccdid", "qid"])

    # if no multiprocessing requested, loop over files serially and load data. return immediately
    if nworkers is None:
        lightcurves = []
        for location in tqdm.tqdm(location_grps):
            lightcurves.append(load_lightcurves_one_file(location))
        return pd.concat(lightcurves, ignore_index=True)

    # if we get here, multiprocessing has been requested

    # make sure the chunksize isn't so big that some workers will sit idle
    if len(location_grps) < nworkers * chunksize:
        chunksize = len(location_grps) // nworkers + 1

    # start a pool of background processes to load data in parallel
    with mp.Pool(nworkers, initializer=helpers.scale_up._init_worker) as pool:
        lightcurves = []
        # use imap because it's lazier than map and can reduce memory usage for long iterables
        # use unordered because we don't care about the order in which results are returned
        # using a large chunksize can make it much faster than the default of 1
        for ztf_df in tqdm.tqdm(
            pool.imap_unordered(load_lightcurves_one_file, location_grps, chunksize=chunksize),
            total=len(location_grps)  # must tell tqdm how many files we're iterating over
        ):
            lightcurves.append(ztf_df)

    return pd.concat(lightcurves, ignore_index=True)


def load_lightcurves_one_file(location):
    """Load light curves from one file.

    Parameters
    ----------
    location : tuple
        tuple containing two elements that describe one parquet file. the elements are:
            location_keys : tuple(str, int, int int)
                Keys that uniquely identify a ZTF location (filtercode, field, CCD, and quadrant).
                Used to lookup the file name.
            location_df : pd.DataFrame
                Dataframe of objects in this location. Used to filter data from this file.

    Returns
    -------
    ztf_df : pd.DataFrame
        Dataframe of light curves. Expect one row per oid in location_df. Each row
        stores a full light curve. Elements in the columns "mag", "hmjd", etc. are arrays.
    """
    location_keys, location_df = location

    # load light curves from this file, filtering for the ZTF object IDs in location_df.
    # we could use pandas or pyarrow. pyarrow plays nicer with multiprocessing. pandas requires 
    # "spawning" new processes, which causes leaked semaphores in some cases.
    ztf_df = pyarrow.parquet.read_table(
        file_name(*location_keys),
        filesystem=pyarrow.fs.S3FileSystem(region="us-east-1"),
        columns=["objectid", "hmjd", "mag", "magerr", "catflags"],
        filters=[("objectid", "in", location_df["oid"].to_list())],
    ).to_pandas()

    # in the parquet files, "objectid" is the ZTF object ID
    # in the MultiIndexDFObject, "objectid" is the ID of a sample_table object
    # rename the ZTF object ID that just got loaded to avoid confusion
    ztf_df = ztf_df.rename(columns={"objectid": "oid"})

    # add the sample_table object ID and label columns by mapping thru the ZTF object ID
    oidmap = location_df.set_index("oid")["objectid"].to_dict()
    lblmap = location_df.set_index("oid")["label"].to_dict()
    ztf_df["objectid"] = ztf_df["oid"].map(oidmap)
    ztf_df["label"] = ztf_df["oid"].map(lblmap)

    # add the band (i.e., filtercode)
    ztf_df["band"] = location_keys[0]

    return ztf_df


def transform_lightcurves(ztf_df):
    """Clean and transform the data into the form expected for a `MultiIndexDFObject`.

    Parameters
    ----------
    ztf_df : pd.DataFrame
        Dataframe of light curves as returned by `load_lightcurves`.

    Returns
    -------
    ztf_df : pd.DataFrame
        The input dataframe, cleaned and transformed.
    """
    # ztf_df might have more than one light curve per (band + objectid) if the ra/dec is close
    # to a CCD-quadrant boundary (Sanchez-Saez et al., 2021). keep the one with the most datapoints
    indexes_to_keep = []
    for _, singleband_object in ztf_df.groupby(["objectid", "band"]):
        if len(singleband_object.index) == 1:
            indexes_to_keep.extend(singleband_object.index)
        else:
            npoints = singleband_object["mag"].str.len()
            npointsmax_object = singleband_object.loc[npoints == npoints.max()]
            # this may still have more than one light curve if they happen to have the same number
            # of datapoints (e.g., Yang sample sample_table[7], band 'zr').
            # arbitrarily pick the one with the min oid.
            # depending on your science, you may want (e.g.,) the largest timespan instead
            if len(npointsmax_object.index) == 1:
                indexes_to_keep.extend(npointsmax_object.index)
            else:
                minoid = npointsmax_object.oid.min()
                indexes_to_keep.extend(npointsmax_object.loc[npointsmax_object.oid == minoid].index)
    ztf_df = ztf_df.loc[sorted(indexes_to_keep)]
    
    # store "hmjd" as "time".
    # note that other light curves in this notebook will have "time" as MJD instead of HMJD.
    # if your science depends on precise times, this will need to be corrected.
    ztf_df = ztf_df.rename(columns={"hmjd": "time"})

    # "explode" the data structure into one row per light curve point and set the correct dtypes
    # note that this operation may require 6+ times the RAM used for ztf_df up to now
    ztf_df = ztf_df.explode(["time", "mag", "magerr", "catflags"], ignore_index=True)
    ztf_df = ztf_df.astype({"time": "float", "mag": "float", "magerr": "float", "catflags": "int"})

    # remove data flagged as bad
    ztf_df = ztf_df.loc[ztf_df["catflags"] < 32768, :]

    # calc flux [https://arxiv.org/pdf/1902.01872.pdf zeropoint corrections already applied]
    mag = ztf_df["mag"].to_numpy()
    magerr = ztf_df["magerr"].to_numpy()
    fluxupper = ((mag - magerr) * u.ABmag).to_value('mJy')
    fluxlower = ((mag + magerr) * u.ABmag).to_value('mJy')
    ztf_df["flux"] = (mag * u.ABmag).to_value('mJy')
    ztf_df["err"] = (fluxupper - fluxlower) / 2

    return ztf_df
