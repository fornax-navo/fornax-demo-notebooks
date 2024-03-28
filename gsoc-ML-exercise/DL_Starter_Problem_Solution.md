---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="--xltAUVwbXR" -->
# GSOC2024 ML/DL Starter Problem
Lucas Martin Garcia
<!-- #endregion -->

<!-- #region id="4PrdvcylBkut" -->
# How to read the data

<!-- #endregion -->

```python 
# How to read the data

import astropy.io.fits as fits
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def abmag(flux):
    return np.where(flux > 0, -2.5 * np.log10(flux) + 23.9, -99.0)

```

<!-- #region id="AFGsJSW6BjTH" -->
## Reading in the input files from Box
<!-- #endregion -->

```python 
## Reading in the input files from Box

## Reading in the 1st field: GOODS-S
url = 'https://caltech.box.com/shared/static/yad6e2mgd1rbngrsfpkod86lly1gip18'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    gs = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download GOODS-S.")

#these are the name of the filters we use in GOODS-S and their central wavelength
centerwave_gs = {'VIMOS_U_FLUX':3734,'ACS_F435W_FLUX':4317,'ACS_F606W_FLUX':5918,'ACS_F775W_FLUX':7617,'ACS_F814W_FLUX':8047,'ACS_F850LP_FLUX':9055,'WFC3_F098M_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'ISAAC_KS_FLUX':21600,'IRAC_CH1_FLUX':36000}

## Reading in the 2nd field: GOODS-N
url = 'https://caltech.box.com/shared/static/jv3gyp0kkxnbql5wnpodujjn4cvchrud'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    gn = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download GOODS-N.")

centerwave_gn = {'KPNO_U_FLUX':3647,'ACS_F435W_FLUX':4317,'ACS_F606W_FLUX':5918,'ACS_F775W_FLUX':7617,'ACS_F814W_FLUX':8047,'ACS_F850LP_FLUX':9055,'WFC3_F105W_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'CFHT_KS_FLUX':21460,'IRAC_CH1_SCANDELS_FLUX':36000}

## Reading in the 3rd field: UDS
url = 'https://caltech.box.com/shared/static/q8oxrb3zisw0xnekrocuydxwoivge91x'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    uds = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download UDS.")


centerwave_uds = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'SUBARU_r_FLUX':5960,'ACS_F606W_FLUX':6500,'SUBARU_i_FLUX':7680,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8890,'HAWKI_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'HAWKI_KS_FLUX':21470,'IRAC_CH1_SEDS_FLUX':36000}

## Reading in the 4th field: EGS
url = 'https://caltech.box.com/shared/static/sthjm6vl6b8bdhvg38lyursps9xnoc6h'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    egs = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download EGS.")

centerwave_egs = {'CFHT_U_FLUX':3825,'CFHT_G_FLUX':4810,'ACS_F606W_FLUX':5960,'CFHT_R_FLUX':6250,'CFHT_I_FLUX':7690,'ACS_F814W_FLUX':8090,'CFHT_Z_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'WIRCAM_K_FLUX':21460,'IRAC_CH1_FLUX':36000}

## Reading in the 5th field: COSMOS
url = 'https://caltech.box.com/shared/static/27xwurf6t3yj2i4mn0atk7gof4y22ow9'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    cos = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download COSMOS.")

centerwave_cos = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'CFHT_G_FLUX':5960,'ACS_F606W_FLUX':6500,'CFHT_I_FLUX':7619,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8829,'ULTRAVISTA_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'ULTRAVISTA_Ks_FLUX':21521,'IRAC_Ch1_FLUX':36000}

```

<!-- #region id="D9YVlL0xBYG5" -->
## Plotting the SED (spectral energy distribution) of one data point
Flux as a function of wavelength
<!-- #endregion -->

```python 
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

<!-- #region id="itaQnfBmBglm" -->

# We need output to be a catalog combined of all 5 fields with flux in the filters below:
<!-- #endregion -->

```python 
centerwave_out = {'u':3826,'g':4877,'r':6231,'i':7618,'z':8828,'y':10214,'j':12535,'h':14453,'k':17316}
```

<!-- #region id="O0QLJQpgF1T3" -->
#Solution
<!-- #endregion -->

<!-- #region id="koLNQrnmWW1i" -->
A simple way to combine all five fields in optical and NIR filters would be to interpolate the flux measures with respect to the wavelengths and obtain the flux directly from the interpolated function. The basic interpolation consists of linear interpolation of the flux points of the galaxies.
This Interpolation process of all the galaxies of all fields requires an estimated execution time of 2~3 minutes.
<!-- #endregion -->

```python 
# Create data list of the galaxy fields
fields = [gs, gn, uds, egs, cos]
centerwave_fields = [centerwave_gs,centerwave_gn,centerwave_uds,centerwave_egs,centerwave_cos]


# Function to interpolate flux measures of all galaxies from an specific field in the centerwave chosen
def interpolate_flux(data_field, centerwave_field, centerwave_out):
    interpolate_results = []
    centerwave_field_values = list(centerwave_field.values())
    centerwave_out_values = list(centerwave_out.values())

    # Apply interpolation to each galaxy
    for id in range(len(data_field)):
        # Flux measures of the current galaxy, filtering out negative flux values
        current_flux = np.array([data_field[filter][id] for filter in centerwave_field])
        valid_flux_indices = current_flux > 0  # Indices of positive flux values
        if not valid_flux_indices.any():
            # If there are no positive flux values, skip this galaxy or handle accordingly
            continue

        # Only use positive flux values and corresponding wavelengths for interpolation
        valid_fluxes = current_flux[valid_flux_indices]
        valid_wavelengths = np.array(centerwave_field_values)[valid_flux_indices]

        # Interpolate the measures to obtain the flux in the wavelengths specified
        interpolated_flux = np.interp(centerwave_out_values, valid_wavelengths, valid_fluxes)
        interpolate_results.append(interpolated_flux)

    return np.array(interpolate_results)

# Apply galaxy interpolation to all fields
combined_fields = []
for i in range(len(fields)):
    current_interpolated_flux = interpolate_flux(fields[i], centerwave_fields[i], centerwave_out)
    combined_fields.append(current_interpolated_flux)

# Convert to numpy
combined_fields = np.vstack(combined_fields)


```

<!-- #region id="CnmKHH_hyF7H" -->
As the result, `combined_fields` would be the one file output with all the galaxies in the requested wavelengths.
<!-- #endregion -->

<!-- #region id="0wE-6lBLxrJ0" -->
As an example we can plot the measurements obtained of the first 3 galaxies in the GOODS-S field with requested wavelengths:
<!-- #endregion -->

```python 
num_galaxies = 3
colors = plt.cm.tab10(np.linspace(0, 1, num_galaxies))
colors2 = plt.cm.viridis(np.linspace(0, 1, num_galaxies))
# Plot first five galaxies
plt.figure(figsize=(8, 4))
for i in range(num_galaxies):
    plt.plot(list(centerwave_out.values()), combined_fields[i],'*',markersize=10,label=f'Galaxy{i+1}',color=colors2[i],markeredgecolor='black',markeredgewidth = 0.5)

plt.xlabel('Wavelength(A)')
plt.ylabel('Flux Interpolated')
plt.legend(loc='upper left')
plt.title('SED of sample galaxies in the GOODS-S field')
plt.show()
```

<!-- #region id="F1cDYn6Cx_4z" -->
An now we can plot the as well the original measurements of the same 3 galaxies with the original wavelengths in addition to the interpolated function from which the new values where obtained to see that the interpolation works well:
<!-- #endregion -->

```python 
num_galaxies = 3
colors = plt.cm.cividis(np.linspace(0, 1, num_galaxies))
colors2 = plt.cm.viridis(np.linspace(0, 1, num_galaxies))
# Plot first five galaxies
plt.figure(figsize=(10, 4))
for i in range(num_galaxies):
    plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], [gs[w][i] for w in centerwave_gs if gs[w][i] > 0], label=f'Interpolation Galaxy {i+1}',color=colors[i])
    plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], [gs[w][i] for w in centerwave_gs if gs[w][i] > 0],'o',markersize=10,color=colors[i],markeredgecolor='black',markeredgewidth = 0.5)
    plt.plot(list(centerwave_out.values()), combined_fields[i],'*',markersize=10,label=f'Galaxy{i+1}',color=colors2[i],markeredgecolor='black',markeredgewidth = 0.5)

plt.xlabel('Wavelength(A)')
plt.ylabel('Flux Interpolated')
plt.legend(loc='upper left')
plt.title('SED of sample galaxies in the GOODS-S field')
plt.show()
```
