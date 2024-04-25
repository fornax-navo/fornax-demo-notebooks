---
jupytext:
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
from astropy.visualization import simple_norm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


gs = fits.getdata('data/gds.fits')
sel1 = (gs['zbest']>0.01)&(gs['zbest']<0.3)&(gs['CLASS_STAR']<0.95)&(gs['Hmag']<24)&(gs['FWHM_IMAGE']>5)
ras, decs = gs['RA_1'][sel1],gs['DEC_1'][sel1]
print(len(ras))
```

***

## 1) cutouts from WISE and Spitzer images

```{code-cell} ipython3
# a coordinate in the GOODS-S field to query for the image urls 
coord = SkyCoord(np.median(ras),np.median(decs), unit='deg')

#To find the collections in irsa 
#Irsa.list_collections()

spitzer_images = Irsa.query_sia(pos=(coord, 15 * u.arcmin), collection='spitzer_scandels').to_table()
science_images = spitzer_images[spitzer_images['dataproduct_subtype'] == 'science']

WISE_images = Irsa.query_sia(pos=(coord, 15 * u.arcmin), collection='wise_allwise').to_table()
wscience_images = WISE_images[WISE_images['dataproduct_subtype'] == 'science']
```

```{code-cell} ipython3
for i in range(4):
    coord = SkyCoord(ras[i],decs[i], unit='deg')

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
                cutout_s = Cutout2D(hdul[0].section, position=coord, size=64, wcs=WCS(hdul[0].header))
                da = np.arcsinh(cutout_s.data)
                p_min, p_max = np.percentile(da, [1, 99])  
                da_clipped = np.clip(da, p_min, p_max)
                norm = simple_norm(da_clipped, 'linear', min_cut=p_min, max_cut=p_max)
                pash = (255 * norm(da_clipped)).astype(np.uint8)
                ax0.imshow(pash,origin='lower')
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
                cutout_w = Cutout2D(hdul[0].section, position=coord, size=64, wcs=WCS(hdul[0].header))
                da = np.arcsinh(cutout_w.data)
                p_min, p_max = np.percentile(da, [5, 95])  
                da_clipped = np.clip(da, p_min, p_max)
                norm = simple_norm(da_clipped, 'linear', min_cut=p_min, max_cut=p_max)
                pash = (255 * norm(da_clipped)).astype(np.uint8)
                ax0.imshow(pash[18:-18,18:-18],origin='lower')
                ax0.text(2,2,str(w['energy_bandpassname']),color='y')
                ax0.axis('off')
            except:
                ax0.text(2,2,str(w['energy_bandpassname']),color='y')
                ax0.axis('off') 
    plt.show()
    
```

## Saving into a hdf5 structure

```{code-cell} ipython3
#%rm 'Sample_train.hdf5'
import h5py
import torchvision.transforms as transforms

sample_size = 500

tfms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_shape = (sample_size, 4, 64, 64)

hdf5_file = h5py.File('Sample_train.hdf5', mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.float32)
hdf5_file.create_dataset("train_labels", (sample_size,), np.float32)
hdf5_file["train_labels"][...] = np.zeros(sample_size)

for i in range(sample_size):
    coord = SkyCoord(ras[i],decs[i], unit='deg')
    pashe = np.zeros((4,64,64))

    if i % 50 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, sample_size))

    for s in science_images:
        with fits.open(s['access_url'], use_fsspec=True) as hdul:
            try:
                cutout_s = Cutout2D(hdul[0].section, position=coord, size=64, wcs=WCS(hdul[0].header))
                da = np.arcsinh(cutout_s.data)
                p_min, p_max = np.percentile(da, [1, 99])  
                da_clipped = np.clip(da, p_min, p_max)
                norm = simple_norm(da_clipped, 'linear', min_cut=p_min, max_cut=p_max)
                pash = (255 * norm(da_clipped)).astype(np.uint8)
                if s['energy_bandpassname']=='IRAC1':
                    pashe[2,:,:] = tfms(pash)
                elif s['energy_bandpassname']=='IRAC2':
                    pashe[3,:,:] = tfms(pash)
                else:
                    continue    
            except:
                continue
    
    for w in wscience_images:
        with fits.open(w['access_url'], use_fsspec=True) as hdul:
            try:
                cutout_w = Cutout2D(hdul[0].section, position=coord, size=64, wcs=WCS(hdul[0].header))
                da = np.arcsinh(cutout_w.data)
                p_min, p_max = np.percentile(da, [1, 99])  
                da_clipped = np.clip(da, p_min, p_max)
                norm = simple_norm(da_clipped, 'linear', min_cut=p_min, max_cut=p_max)
                pash = (255 * norm(da_clipped)).astype(np.uint8)
                if w['energy_bandpassname']=='W1':
                    pashe[0,:,:] = tfms(pash)
                elif w['energy_bandpassname']=='W2':
                    pashe[1,:,:] = tfms(pash)
                else:
                    continue
            except:
                continue
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = pashe

hdf5_file.close()
```

## Loading the hdf5

```{code-cell} ipython3
import torch
from galaxy_hdf5loader import galaxydata

dataset = galaxydata('Sample_train.hdf5')
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=20,shuffle=True, num_workers=int(0))

inputs, classes = next(iter(dataloader))  
real_cpu = inputs.to('cpu')
ajab = real_cpu.detach()
ajab = ajab.cpu()
```

```{code-cell} ipython3
k=np.random.randint(20)
plt.figure(figsize=(14,3))
for i in range(2):
    plt.subplot(1,4,i+1) 
    plt.imshow(ajab[k,i,18:-18,18:-18],origin='lower')
    plt.colorbar()
    plt.axis('off')

for i in range(2,4):
    plt.subplot(1,4,i+1) 
    plt.imshow(ajab[k,i,:,:],origin='lower')
    plt.colorbar()
    plt.axis('off')
plt.tight_layout()
```

```{code-cell} ipython3

```
