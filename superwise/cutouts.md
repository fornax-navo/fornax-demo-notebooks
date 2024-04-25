---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: root *
  language: python
  name: conda-root-py
---

# Super WISE: To Enhance WISE images learning from Spitzer

By the IPAC Science Platform Team, started: Apr 24, 2024- last edit: Apr 24, 2024

***

```{code-cell} ipython3
#!pip install -r requirements.txt

import sys
sys.path.append('code_src/')

from astropy.table import Table
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.ipac.irsa import Irsa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read from the GOODS-S catalog, to big for fornax! 
ras = [53.1016865, 53.0884484, 53.0720028, 53.0990123, 53.1462097, 53.1540879,
    53.1143036, 53.1641068, 53.1461636, 53.2072268, 53.1503938, 53.2076163,
    53.09177, 53.1021871, 53.1362766, 53.1454835, 53.2141874, 53.2073151,
    53.1894621, 53.2103125, 53.1842558, 53.242204, 53.1048801, 53.1351058]
decs = [
    -27.9584786, -27.9341995, -27.9324571, -27.9245559, -27.9245272, -27.9233587,
    -27.9223383, -27.9186358, -27.9258091, -27.9161896, -27.9169205, -27.9141329,
    -27.9078953, -27.9119265, -27.9063466, -27.9038478, -27.9054073, -27.901093,
    -27.9042908, -27.9057874, -27.899627, -27.8988661, -27.9009879, -27.8951824]
```

***

## 1) cutouts from WISE and Spitzer images

```{code-cell} ipython3
# a coordinate in the GOODS-S field to query for the image urls 
coord = SkyCoord(ras[0],decs[0], unit='deg')

#To find the collections in irsa 
#from astroquery.ipac.irsa import Irsa
#Irsa.list_collections()

spitzer_images = Irsa.query_sia(pos=(coord, 15 * u.arcmin), collection='spitzer_scandels').to_table()
science_images = spitzer_images[spitzer_images['dataproduct_subtype'] == 'science']

WISE_images = Irsa.query_sia(pos=(coord, 15 * u.arcmin), collection='wise_allwise').to_table()
wscience_images = WISE_images[WISE_images['dataproduct_subtype'] == 'science']
```

```{code-cell} ipython3
for r,d in zip(ras,decs):
    coord = SkyCoord(r,d, unit='deg')

    plt.figure(figsize=(10,3))
    for s in science_images:
        if s['energy_bandpassname']=='IRAC1':
            ax0 = plt.subplot(1,4,3)
        elif s['energy_bandpassname']=='IRAC2':
            ax0 = plt.subplot(1,4,4)
        else:
            continue    
        with fits.open(s['access_url'], use_fsspec=True) as hdul:
            try:
                cutout_s = Cutout2D(hdul[0].section, position=coord, size=25 * u.arcsec, wcs=WCS(hdul[0].header))
                ax0.imshow(cutout_s.data,origin='lower')
                ax0.text(2,2,str(s['energy_bandpassname']),color='y')
                ax0.axis('off')
            except:
                ax0.text(2,2,str(s['energy_bandpassname']),color='y')
                ax0.axis('off') 

    for w in wscience_images:
        if w['energy_bandpassname']=='W1':
            ax0 = plt.subplot(1,4,1)
        elif w['energy_bandpassname']=='W2':
            ax0 = plt.subplot(1,4,2)
        else:
            continue
        
        with fits.open(w['access_url'], use_fsspec=True) as hdul:
            try:
                cutout_w = Cutout2D(hdul[0].section, position=coord, size=25 * u.arcsec, wcs=WCS(hdul[0].header))
                ax0.imshow(cutout_w.data,origin='lower')
                ax0.text(2,2,str(w['energy_bandpassname']),color='y')
                ax0.axis('off')
            except:
                ax0.text(2,2,str(w['energy_bandpassname']),color='y')
                ax0.axis('off') 
    plt.show()
```

```{code-cell} ipython3

```
