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

```{code-cell} ipython3
import astropy.io.fits as fits
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def abmag(flux):
    return np.where(flux > 0, -2.5 * np.log10(flux) + 23.9, -99.0)
```

# Reading in the input files from Box

```{code-cell} ipython3
## Reading in the 1st field: GOODS-S 
url = 'https://caltech.box.com/shared/static/yad6e2mgd1rbngrsfpkod86lly1gip18'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    gs = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download GOODS-S.")
    
#these are the name of the filters we use in GOODS-S and their central wavelength
centerwave_gs = {'VIMOS_U_FLUX':3734,'ACS_F435W_FLUX':4317,'ACS_F606W_FLUX':5918,'ACS_F775W_FLUX':7617,'ACS_F814W_FLUX':8047,'ACS_F850LP_FLUX':9055,'WFC3_F098M_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'ISAAC_KS_FLUX':21600,'IRAC_CH1_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 2nd field: GOODS-N 
url = 'https://caltech.box.com/shared/static/jv3gyp0kkxnbql5wnpodujjn4cvchrud'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    gn = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download GOODS-N.")
    
centerwave_gn = {'KPNO_U_FLUX':3647,'ACS_F435W_FLUX':4317,'ACS_F606W_FLUX':5918,'ACS_F775W_FLUX':7617,'ACS_F814W_FLUX':8047,'ACS_F850LP_FLUX':9055,'WFC3_F105W_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'CFHT_KS_FLUX':21460,'IRAC_CH1_SCANDELS_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 3rd field: UDS 
url = 'https://caltech.box.com/shared/static/q8oxrb3zisw0xnekrocuydxwoivge91x'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    uds = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download UDS.")
    

centerwave_uds = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'SUBARU_r_FLUX':5960,'ACS_F606W_FLUX':6500,'SUBARU_i_FLUX':7680,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8890,'HAWKI_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'HAWKI_KS_FLUX':21470,'IRAC_CH1_SEDS_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 4th field: EGS 
url = 'https://caltech.box.com/shared/static/sthjm6vl6b8bdhvg38lyursps9xnoc6h'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    egs = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download EGS.")
    
centerwave_egs = {'CFHT_U_FLUX':3825,'CFHT_G_FLUX':4810,'ACS_F606W_FLUX':5960,'CFHT_R_FLUX':6250,'CFHT_I_FLUX':7690,'ACS_F814W_FLUX':8090,'CFHT_Z_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'WIRCAM_K_FLUX':21460,'IRAC_CH1_FLUX':36000}
```

```{code-cell} ipython3
## Reading in the 5th field: COSMOS
url = 'https://caltech.box.com/shared/static/27xwurf6t3yj2i4mn0atk7gof4y22ow9'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    cos = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download COSMOS.")
    
centerwave_cos = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'CFHT_G_FLUX':5960,'ACS_F606W_FLUX':6500,'CFHT_I_FLUX':7619,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8829,'ULTRAVISTA_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'ULTRAVISTA_Ks_FLUX':21521,'IRAC_Ch1_FLUX':36000}
```

# Plotting the SED (spectral energy distribution) of one data point

```{code-cell} ipython3
randomid = np.random.randint(len(gs))
plt.figure(figsize=(6,4))
plt.title('SED of a galaxy in the GOODS-S field')
for w in centerwave_gs:
    if gs[w][randomid] > 0:  # Only plot positive flux values
        plt.plot(centerwave_gs[w],gs[w][randomid],'r*',markersize=10)

plt.yscale('log')
plt.xlabel('Wavelength(A)')
plt.ylabel('Flux (microJansky)')
```

# We need output to be a catalog combined of all 5 fields with flux in the filters below:

```{code-cell} ipython3
centerwave_out = {'u':3826, 'g':4877, 'r':6231, 'i':7618, 'z':8828, 'y':10214, 'j':12535, 'h':14453, 'k':17316}
```

# Solution 

```{code-cell} ipython3
# extract required wavelengths in a list
centerwave_out_val = list(centerwave_out.values())
```

```{code-cell} ipython3
# a function that extract the positive flux values and correponding wavelengths only
def extract_positives(flux, wave):
    flux_pos = [flx for flx in flux if flx > 0]
    corresponding_centerwave = [wave[index] for index, flx in enumerate(flux) if flx > 0]
    return flux_pos, corresponding_centerwave
```

```{code-cell} ipython3
from scipy.interpolate import interp1d

# interpolation
def interpolate_data(data, centerwave, interp_kind):
  f = interp1d(centerwave, data, interp_kind, fill_value='extrapolate')
  return f(centerwave_out_val)
```

```{code-cell} ipython3
results = {} # the combined catalog

for field_name, field_data, centerwave in [
    ('GOODS-S', gs, centerwave_gs),
    ('GOODS-N', gn, centerwave_gn),
    ('UDS', uds, centerwave_uds),
    ('EGS', egs, centerwave_egs),
    ('COSMOS', cos, centerwave_cos)
]:
    flux_in = []
    for w in centerwave:
        flux_in.append(field_data[w])
    flux_in = np.array(flux_in)
    
    centerwave_in = list(centerwave.values())
    
    flux_out = []
    for galx in range(flux_in.shape[1]):
        positive_fluxes, corresp_centerwave = extract_positives(flux_in[:, galx], centerwave_in)
        if len(positive_fluxes) > 0:
            interp = interpolate_data(positive_fluxes, corresp_centerwave, 'linear')
        else:
            interp = np.zeros(len(centerwave_out_val))
        flux_out.append(interp)
    
    results[field_name] = np.array(flux_out)
```

# Plotting the results

```{code-cell} ipython3
randomid = np.random.randint(len(gs))
plt.figure(figsize=(6,4))
plt.title('SED of a galaxy in the GOODS-S field')

centerwave_gs_vis = []
gs_vis = []
for w in centerwave_gs:
    if gs[w][randomid] > 0:  # Only plot positive flux values
        centerwave_gs_vis.append(centerwave_gs[w])
        gs_vis.append(gs[w][randomid])

plt.plot(centerwave_gs_vis, gs_vis,'r*',markersize=10, label='Original')
plt.plot(centerwave_out_val, results['GOODS-S'][randomid], 'g*', label='Intepolated')
plt.legend()

plt.yscale('log')
plt.xlabel('Wavelength(A)')
plt.ylabel('Flux (microJansky)');
```
