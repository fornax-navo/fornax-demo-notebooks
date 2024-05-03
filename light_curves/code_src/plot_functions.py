import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy import stats


# List bands that need special treatment
ELECTRON_BANDS = ["K2", "Kepler", "TESS"]  # flux in electrons/sec
ICECUBE_BAND = "IceCube"  # "fluxes" are just events (= limits on plot)
ZTF_BANDS = ["zg", "zr", "zi"]  # will be plotted in "B" zoom-in for better visibility
# List all other bands
OTHER_BANDS = ["F814W",  # HCV
               "FERMIGTRIG", "SAXGRBMGRB",  # HEASARC
               "G", "BP", "RP",  # Gaia
               "panstarrs g", "panstarrs i", "panstarrs r", "panstarrs y", "panstarrs z",  # Pan-STARRS
               "W1", "W2"]  # WISE

# Set a color for each band.
# Create a generator to yield colors (this will be useful if we need to plot bands not listed above).
COLORS = (color for color in mpl.colormaps["tab20"].colors + mpl.colormaps["tab20b"].colors)
BAND_COLORS = {band: next(COLORS) for band in [ICECUBE_BAND] + ZTF_BANDS + ELECTRON_BANDS + OTHER_BANDS}


def create_figures(df_lc, show_nbr_figures, save_output):
    '''
    Creates figures of the lightcurves for each object in df_lc.
    
    Parameters
    ----------
    df_lc : lightcurve object
        Lightcurve objects from which to create the lightcurve figures.
        
    show_nbr_figures : int
        Number of figures to show inline. For example, `show_nbr_figures = 5' would
        show the first 5 figures inline.
        
    save_output: bool
        Whether to save the lightcurve figures. If saved, they will be in the "output" directory.
        
    
    Returns
    -------
    Saves the figures in the output directory
    
    
    Notes
    -----
    By default, if save_output is True, figures are
    made and saved for *all* sources. If that is too much, the user can create a selection
    from the sample of which sources they like to plot (and save).
    
    '''
    
    if (show_nbr_figures == 0) & (not save_output):
        print("No figures are shown or saved.")
        return(False)
    
    # Iterate over objects and create figures
    for cc, (objectid, singleobj_df) in enumerate(df_lc.data.groupby('objectid')):
        # Set up for plotting. We use the "mosaic" method so we can plot
        # the ZTF data in a subplot for better visibility.
        fig, axes = plt.subplot_mosaic(mosaic=[["A"],["A"],["B"]] , figsize=(10,8))

        # Iterate over bands and plot light curves.
        # IceCube needs to be done last so that we know the y-axis limits.
        band_groups = _clean_lightcurves(singleobj_df).groupby('band')
        max_fluxes = band_groups.flux.max()  # max flux per band
        for band, band_df in band_groups:
            if band == ICECUBE_BAND:
                continue
            _plot_lightcurve(band, band_df, axes, max_fluxes)
        if ICECUBE_BAND in band_groups.groups:
            _plot_lightcurve(ICECUBE_BAND, band_groups.get_group(ICECUBE_BAND), axes)

        # Format the figure.
        # Get the ZTF min/max times for the zoom in (will be NaN if there are no ZTF bands).
        ztf_df = band_groups.filter(lambda band_df: band_df.name in ZTF_BANDS)
        _format_axes(axes, ztf_time_min_max=(ztf_df.time.min(), ztf_df.time.max()))
        plt.subplots_adjust(hspace=0.3 , wspace=0.3)
        axes["A"].legend(bbox_to_anchor=(1.2,0.95), title=f"objectid: {objectid}")

        # Save, show, and/or close the figure.
        if save_output:
            savename = os.path.join("output" , "lightcurve_{}.pdf".format(objectid) ) 
            plt.savefig(savename, bbox_inches="tight")
        if cc < show_nbr_figures:
            plt.show()
        else:
            plt.close()
            # If figures are not saved, we only have to loop over the number of figures shown.
            if not save_output:
                break
            
    print("Done")
    return(True)
    
def create_figure(df_lc, index, save_output):
    '''
    Creates a figure of the lightcurve for a specific object in df_lc, identified by index.
    
    Parameters
    ----------
    df_lc : DataFrame
        DataFrame containing lightcurve objects from which to create the lightcurve figure.
        
    index : int
        Index of the object to plot. This is not the object ID but the position in the list of grouped objects.
        
    save_output: bool
        Whether to save the lightcurve figure. If saved, it will be in the "output" directory.
        
    
    Returns
    -------
    Saves the figure in the output directory if save_output is True.
    Shows the figure inline if save_output is False.
    
    '''
    
    grouped = list(df_lc.groupby('objectid'))
    if index < 0 or index >= len(grouped):
        print(f"Index {index} is out of bounds. No figure created.")
        return False
    
    objectid, singleobj_df = grouped[index]
    
    # Set up for plotting.
    fig, axes = plt.subplot_mosaic(mosaic=[["A"],["A"],["B"]] , figsize=(10,6))

    # Iterate over bands and plot light curves.
    band_groups = _clean_lightcurves(singleobj_df).groupby('band')
    max_fluxes = band_groups.flux.max()  # max flux per band
    for band, band_df in band_groups:
        if band == ICECUBE_BAND:
            continue
        _plot_lightcurve(band, band_df, axes, max_fluxes)
    if ICECUBE_BAND in band_groups.groups:
        _plot_lightcurve(ICECUBE_BAND, band_groups.get_group(ICECUBE_BAND), axes)

    # Format the figure.
    ztf_df = band_groups.filter(lambda band_df: band_df.name in ZTF_BANDS)
    _format_axes(axes, ztf_time_min_max=(ztf_df.time.min(), ztf_df.time.max()))
    plt.subplots_adjust(hspace=0.2 , wspace=0.3)
    axes["A"].legend()

    # Save or show the figure.
    if save_output:
        savename = os.path.join("output" , f"lightcurve_{objectid}.png")
        plt.savefig(savename, bbox_inches="tight")
        print(f"Figure saved as {savename}")
    else:
        plt.show()

    plt.close()
    return True


def _clean_lightcurves(singleobj_df):
    """Clean the dataframe in preparation for plotting.

    Parameters
    ----------
    singleobj_df : pandas.DataFrame
        DataFrame for a single object containing light curves in one or more bands.

    Returns
    -------
    singleobj_df cleaned of "bad" rows and reformatted for plotting.
    """
    # Sort so that time increases monotonically within each band.
    # Switch the index to columns (reset) so they can be plotted.
    singleobj = singleobj_df.sort_index().reset_index()

    # Remove rows containing NaN in time, flux, or err
    singleobj = singleobj.dropna(subset=["time", "flux", "err"])

    # Do sigma-clipping per band.
    band_groups = singleobj.groupby("band").flux
    zscore = band_groups.transform(lambda fluxes: np.abs(stats.zscore(fluxes)))
    n_points = band_groups.transform("size")  # number of data points in the band
    # Keep data points with a zscore < 3 or in a band with less than 10 data points.
    singleobj = singleobj[(zscore < 3.0) | (n_points < 10)]

    return singleobj


def _plot_lightcurve(band, band_df, axes, max_fluxes=None):
    """Plot the single-band light curve.

    Parameters
    ----------
    band : str
        Name of the band.
    band_df : pandas.DataFrame
        Single-band light curve.
    axes : matplotlib.axes.Axes
        Axes object to plot the light curve in.
    max_fluxes : pd.Series
        Maximum flux per band. Used to scale the bands with fluxes in electrons/sec.
    """
    # Plot the light curve. Some bands need special treatment.
    # Get the band color, or the next color in the generator if the band isn't listed above.
    color = BAND_COLORS.get(band) or next(COLORS)

    if band in ELECTRON_BANDS:
        # Fluxes are in electrons/sec. Scale them to the other available fluxes.
        mean_max_flux = max_fluxes[[b for b in max_fluxes.index if b not in ELECTRON_BANDS + [ICECUBE_BAND]]].mean()
        max_electrons = band_df.flux.max()
        factor = mean_max_flux / max_electrons
        axes["A"].errorbar(band_df.time, band_df.flux * factor, band_df.err * factor,
                           capsize=3.0, label=band, color=color)

    elif band == ICECUBE_BAND:
        # "Fluxes" are actually just events (= limits on plot).
        y = axes["A"].get_ylim()[0] + np.diff(axes["A"].get_ylim()) * 0.7
        dy = np.diff(axes["A"].get_ylim()) / 20
        axes["A"].errorbar(band_df.time, np.repeat(y, len(band_df.time)), yerr=dy, uplims=True,
                           fmt="o", label=band, color=color)

    elif band in ZTF_BANDS:
        # ZTF is special because we are plotting the data also in "B" zoom-in.
        _plot_ztf_lightcurve(band, band_df, axes)

    else:
        axes["A"].errorbar(band_df.time, band_df.flux, band_df.err, capsize=3.0, label=band, color=color)


def _plot_ztf_lightcurve(band, band_df, axes):
    """Plot the ZTF single-band light curve.

    Parameters
    ----------
    band : str
        Name of the band.
    band_df : pandas.DataFrame
        Single-band light curve.
    axes : matplotlib.axes.Axes
        Axes object to plot the light curve in.
    """
    color = BAND_COLORS.get(band) or next(COLORS)

    # Plot on "A" axis.
    axes["A"].errorbar(band_df.time, band_df.flux, band_df.err, capsize=1.0, elinewidth=0.5,
                            marker="o", markersize=2, linestyle="", label=f"ZTF {band}", color=color)

    # Plot "B" zoomin
    axes["B"].errorbar(band_df.time, band_df.flux, band_df.err, capsize=1.0, elinewidth=0.5,
                       marker="o", linestyle="", markersize=0.5, alpha=0.5, label=f"ZTF {band}", color=color)

    # Overplot running mean for ZTF in zoomin. Use .values to remove indexing.
    xx = band_df.time.values
    yy = band_df.flux.values
    ee = band_df.err.values
    x_bin = 30  # in MJD
    x_grid = np.arange(np.nanmin(xx), np.nanmax(xx) + x_bin / 4, x_bin / 4)
    tmp = Table(names=["xbin", "ybin", "yerr"])

    for xxx in x_grid:
        s = np.where(np.abs(xx - xxx) < x_bin / 2)[0]
        if len(s) > 1:
            mn = np.nansum(yy[s] * ee[s]) / np.nansum(ee[s])  # weighted mean
            tmp.add_row([xxx, mn, np.nanstd(yy[s])])
        else:
            tmp.add_row([xxx, np.nan, np.nan])

    axes["B"].plot(tmp["xbin"], tmp["ybin"], "-", linewidth=1.5, color=color)


def _format_axes(axes, ztf_time_min_max):
    """Format the axes to look nice.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Axes object to plot the light curve in.
    ztf_time_min_max : tuple containing two floats
        Min and max times of ZTF observations. Used to set x-axis limits on "B" zoomin. If there are no ZTF bands
        in the figure, skip this by sending None or a tuple of NaNs.
    """
    # Set axes limits and labels
    axes["A"].set_ylabel("Flux(mJy)")
    # If we have ZTF limits, set labels and limits on the "B" zoomin.
    if ztf_time_min_max and not any(np.isnan(ztf_time_min_max)):
        axes["B"].set_ylabel("Flux(mJy)")
        axes["B"].set_xlabel("Time(MJD)")
        axes["B"].set_xlim(ztf_time_min_max[0] - 100, ztf_time_min_max[1] + 100)
    else:
        axes["A"].set_xlabel("Time(MJD)")

    ## Set axes ticks and grids
    axes["A"].grid(linestyle=":", color="lightgray", linewidth=1)
    axes["A"].minorticks_on()
    axes["A"].tick_params(axis="x", which="minor", bottom=True)
    axes["A"].tick_params(axis="both", which="major", direction="in", length=6, width=1)
    axes["A"].tick_params(axis="both", which="minor", direction="in", length=3, width=1)
    axes["B"].grid(linestyle=":", color="lightgray", linewidth=1)
    axes["B"].minorticks_on()
    axes["B"].tick_params(axis="x", which="minor", bottom=True)
    axes["B"].tick_params(axis="both", which="major", direction="in", length=6, width=1)
    axes["B"].tick_params(axis="both", which="minor", direction="in", length=3, width=1)
