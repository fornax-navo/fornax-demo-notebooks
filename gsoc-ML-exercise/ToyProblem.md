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

# Installing Necessary packages

```{code-cell} ipython3
# io is already installed along with python for input-output handling

# !conda create --name NasaDL Create a separate conda environment so as to maintain unifromity and organize packages well
# !conda activate NasaDL
# !conda install astropy requests numpy matplotlib scikit-learn pytorch -y 
```

# How to read the data

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

+++

Flux as a function of wavelength

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
centerwave_out = {'u':3826,'g':4877,'r':6231,'i':7618,'z':8828,'y':10214,'j':12535,'h':14453,'k':17316}
```

```{code-cell} ipython3
# Create a list for the galaxy fields and their respective centerwave values
fields = [gs, gn, uds, egs, cos]
centerwave_fields = [centerwave_gs, centerwave_gn, centerwave_uds, centerwave_egs, centerwave_cos]
```

# Solution

+++

# Observations:

```{code-cell} ipython3
# The first thing I notices was that the functon plotting the curves was using random numbers for plots:
# this imples that there's a something that needs to be unveiled in the fetched data
# randomid = np.random.randint(len(gs))
plt.figure(figsize=(6,4))
plt.title('SED of a galaxy in the GOODS-S field')
for w in centerwave_gs:
    for indx in range(int(len(gs)/100)):  # divided by 100 to decrease the number of samples to be plotted =>  hence decrease in time taken to plot 
      if gs[w][indx] > 0:  # Only plot positive flux values
          plt.plot(centerwave_gs[w],gs[w][indx],'r*',markersize=10)

plt.yscale('log')
plt.xlabel('Wavelength(A)')
plt.ylabel('Flux (microJansky)')
```

This implies that the there are many values of Flux for a single value of the wavelength depending on galaxies choosen.

+++

The second observation was to predict which is the best interpolating function to be choosen amongst the vast options available like:
1) Linear Interpolation
2) Polynomial Interpolation
3) Regression ML models ( as it is used to determine functions with continuous value output with higher accuracies )
4) SVM ( Support Vector Machine)

I went on to try some of them as shown below:

+++

# Utility Functions

```{code-cell} ipython3
import matplotlib.pyplot as plt

def plot_sed(combined_fields, centerwave_out, num_galaxies=5):
    """
    Plots the SED (Spectral Energy Distribution) for sample galaxies in a field.

    Args:
    - combined_fields: Numpy array containing flux measures for galaxies.
    - centerwave_out: Dictionary containing output wavelengths.
    - num_galaxies: Number of galaxies to plot.
    """
    color_map = plt.cm.tab10(np.linspace(0, 1, num_galaxies))

    plt.figure(figsize=(8, 4))
    for i in range(num_galaxies):
        plt.plot(list(centerwave_out.values()), combined_fields[i], '*', markersize=10, label=f'Galaxy {i+1}', color=color_map[i], markeredgecolor='black', markeredgewidth=0.5)

    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux Interpolated')
    plt.legend(loc='upper left')
    plt.title('SED of sample galaxies')
    plt.show()
```

```{code-cell} ipython3
def plot_prediction(combined_fields, centerwave_out, centerwave_field=centerwave_gs, field=gs, num_galaxies=5):
    color_map1 = plt.cm.tab10(np.linspace(0, 1, num_galaxies))
    color_map2 = plt.cm.Set2(np.linspace(0, 1, num_galaxies))
    marker_styles = ['o', 's', '^', 'D', 'x']  # Different marker styles for each galaxy

    # Plot first five galaxies
    plt.figure(figsize=(10, 4))
    for i in range(num_galaxies):
        # Plot interpolated flux measures for each filter
        plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], 
                 [gs[w][i] for w in centerwave_gs if gs[w][i] > 0], 
                 label=f'Interpolation Galaxy {i+1}', color=color_map1[i], linestyle='-')
        
        # Plot markers for interpolated flux measures
        plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], 
                 [gs[w][i] for w in centerwave_gs if gs[w][i] > 0],
                 marker=marker_styles[i], markersize=8, color=color_map1[i], markeredgecolor='black', markeredgewidth=0.5)

        # Plot actual interpolated flux measures
        plt.plot(list(centerwave_out.values()), combined_fields[i], 
                 marker=marker_styles[i], markersize=10, label=f'Galaxy {i+1}', color=color_map2[i], markeredgecolor='black', markeredgewidth=0.5)

    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux Interpolated')
    plt.legend(loc='upper left')
    plt.title('SED of sample galaxies in the GOODS-S field')
    plt.show()
```

# Linear Interpolation

```{code-cell} ipython3
# Function to interpolate flux measures of galaxies from a specific field
def linear_interpolate_flux(data_field, centerwave_field, centerwave_out):
    """
    Interpolates flux measures of all galaxies from a specific field to the desired wavelengths.

    Args:
    - data_field: Dictionary containing flux measures for each galaxy in the field.
    - centerwave_field: Dictionary containing central wavelengths for each filter in the field.
    - centerwave_out: Dictionary containing desired output wavelengths.

    Returns:
    - Numpy array of interpolated flux values for all galaxies.
    """
    interpolated_fluxes = []

    # Extract wavelength values from dictionaries
    centerwave_field_values = list(centerwave_field.values())
    centerwave_out_values = list(centerwave_out.values())

    # Iterate through galaxies in the field
    for galaxy_data in range(len(data_field)):
        # Filter out negative flux values
        # valid_flux_indices = [int(flux) > 0 for flux in galaxy_data] 
        current_flux = np.array([data_field[filter][galaxy_data] for filter in centerwave_field])
        valid_flux_indices = current_flux > 0  # Indices of positive flux values
        if not valid_flux_indices.any():
            # Skip galaxy if no positive flux values
            continue

        # Use positive flux values and corresponding wavelengths for interpolation
        valid_fluxes = current_flux[valid_flux_indices]
        valid_wavelengths = np.array(centerwave_field_values)[valid_flux_indices]

        # Interpolate flux values to obtain flux in requested wavelengths
        interpolated_flux = np.interp(centerwave_out_values, valid_wavelengths, valid_fluxes)
        interpolated_fluxes.append(interpolated_flux)

    return np.array(interpolated_fluxes)
```

```{code-cell} ipython3
# Interpolate flux for galaxies in the field
combined_fields = []

# Loop over the 5 fields and append the values to the list
for i in range(len(fields)):
    current_interpolated_flux = linear_interpolate_flux(fields[i], centerwave_fields[i], centerwave_out)
    combined_fields.append(current_interpolated_flux)

# Convert to numpy
combined_fields = np.vstack(combined_fields)
```

```{code-cell} ipython3
# Plot SED of sample galaxies
plot_sed(combined_fields, centerwave_out)
```

```{code-cell} ipython3
# Interpolation of new galaxies based on previous data points 
plot_prediction(combined_fields, centerwave_out)
```

The interpolation defined linearlly seems to fit well to the real curve

+++

# Polynomial Interpolation

```{code-cell} ipython3
def polynomial_interpolate_flux(data_field, centerwave_field, centerwave_out, degree=3):
    """
    Interpolate flux measures of all galaxies from a specific field at the chosen centerwave.

    Args:
    - data_field (dict): Dictionary containing flux measures for each filter of a galaxy.
    - centerwave_field (dict): Dictionary containing center wavelengths for each filter.
    - centerwave_out (dict): Dictionary containing output center wavelengths for interpolation.
    - degree (int): Degree of polynomial for interpolation. Default is 3.

    Returns:
    - np.ndarray: Interpolated flux measures for all galaxies.
    """
    interpolated_fluxes = []

    # Extracting wavelength and flux values
    centerwave_field_values = list(centerwave_field.values())
    centerwave_out_values = list(centerwave_out.values())

    # Iterate over each galaxy in the field
    for galaxy_id in range(len(data_field)):
        # Extract flux measures for the current galaxy, filtering out negative flux values
        current_fluxes = np.array([data_field[filter_name][galaxy_id] for filter_name in centerwave_field])
        valid_flux_indices = current_fluxes > 0

        # Check if there are any positive flux values
        if not valid_flux_indices.any():
            # If no positive flux values, skip this galaxy
            continue

        # Filter out negative flux values and corresponding wavelengths
        valid_fluxes = current_fluxes[valid_flux_indices]
        valid_wavelengths = np.array(centerwave_field_values)[valid_flux_indices]

        # Perform polynomial interpolation
        coefficients = np.polyfit(valid_wavelengths, valid_fluxes, degree)
        interpolated_flux = np.polyval(coefficients, centerwave_out_values)

        # Append interpolated flux for the current galaxy
        interpolated_fluxes.append(interpolated_flux)

    return np.array(interpolated_fluxes)
```

```{code-cell} ipython3
# Example usage:
degree = 3  # Degree of polynomial for interpolation, can be taken as any arbitrary number for sampling

# Apply galaxy interpolation to all fields
combined_fields = []
for field_index in range(len(fields)):
    current_interpolated_flux = polynomial_interpolate_flux(fields[field_index], centerwave_fields[field_index], centerwave_out, degree)
    combined_fields.append(current_interpolated_flux)

# Convert to numpy array
combined_fields = np.vstack(combined_fields)
```

```{code-cell} ipython3
# Plot SED of sample galaxies
plot_sed(combined_fields, centerwave_out)
```

```{code-cell} ipython3
# Interpolation of new galaxies based on previous data points 
plot_prediction(combined_fields, centerwave_out)
plt.show()
```

The interpolation defined with an higher degree polynomial seems to fit way better than linear interpolation to the real curve

+++

# Deep learning Interpolation

```{code-cell} ipython3
import torch

def deepNN_interpolate_flux(data_field, centerwave_field, centerwave_out):
    interpolated_fluxes = []

    centerwave_field_values = list(centerwave_field.values())
    centerwave_out_values = list(centerwave_out.values())

    # Iterate through galaxies in the field
    for galaxy_data in range(len(data_field)):
        current_flux = np.array([data_field[filter][galaxy_data] for filter in centerwave_field])
        valid_flux_indices = current_flux > 0  
        if not valid_flux_indices.any():
            continue

        valid_fluxes = current_flux[valid_flux_indices]
        valid_wavelengths = np.array(centerwave_field_values)[valid_flux_indices]

        # Linear interpolation using PyTorch
        interpolated_flux = torch.tensor(np.interp(centerwave_out_values, valid_wavelengths, valid_fluxes))
        interpolated_fluxes.append(interpolated_flux)

    return torch.stack(interpolated_fluxes)
```

```{code-cell} ipython3
# Apply galaxy interpolation to all fields
combined_fields = []
for i in range(len(fields)):
    current_interpolated_flux = deepNN_interpolate_flux(fields[i], centerwave_fields[i], centerwave_out)
    combined_fields.append(current_interpolated_flux)

# Convert to numpy
combined_fields = np.vstack(combined_fields)
```

```{code-cell} ipython3
# Plot SED of sample galaxies
plot_sed(combined_fields, centerwave_out)
```

```{code-cell} ipython3
# Interpolation of new galaxies based on previous data points 
plot_prediction(combined_fields, centerwave_out)
```

This deep Neural Netwrok architecture for interpolation is my favourite because of the following two reasons:
1) Very fast ~ 1 min of training required
2) Capable of accurately predicting each and every data point

+++

# Support Vector Machine Interpolation

```{code-cell} ipython3
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ml_interpolate_flux(data_field, centerwave_field, centerwave_out):
    """
    Interpolates flux measures of all galaxies from a specific field to the desired wavelengths
    using Support Vector Regression (SVR) machine learning algorithm.

    Args:
    - data_field: Dictionary containing flux measures for each galaxy in the field.
    - centerwave_field: Dictionary containing central wavelengths for each filter in the field.
    - centerwave_out: Dictionary containing desired output wavelengths.

    Returns:
    - Numpy array of interpolated flux values for all galaxies.
    """
    interpolated_fluxes = []

    # Extract wavelength values from dictionaries
    centerwave_field_values = list(centerwave_field.values())
    centerwave_out_values = list(centerwave_out.values())

    # Iterate through galaxies in the field
    for galaxy_data in range(len(data_field)):
        # Filter out negative flux values
        current_flux = np.array([data_field[filter][galaxy_data] for filter in centerwave_field])
        valid_flux_indices = current_flux > 0  # Indices of positive flux values
        if not valid_flux_indices.any():
            # Skip galaxy if no positive flux values
            continue

        # Use positive flux values and corresponding wavelengths for interpolation
        valid_fluxes = current_flux[valid_flux_indices]
        valid_wavelengths = np.array(centerwave_field_values)[valid_flux_indices]

        # Support Vector Regression (SVR) model
        svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1))
        svr_model.fit(valid_wavelengths.reshape(-1, 1), valid_fluxes)

        # Interpolate flux values to obtain flux in requested wavelengths using SVR model
        interpolated_flux = svr_model.predict(np.array(centerwave_out_values).reshape(-1, 1))
        interpolated_fluxes.append(interpolated_flux)

    return np.array(interpolated_fluxes)
```

```{code-cell} ipython3
# Example usage:

# Apply galaxy interpolation to all fields
combined_fields = []
for field_index in range(len(fields)):
    current_interpolated_flux = ml_interpolate_flux(fields[field_index], centerwave_fields[field_index], centerwave_out)
    combined_fields.append(current_interpolated_flux)

# Convert to numpy array
combined_fields = np.vstack(combined_fields)
```

```{code-cell} ipython3
# Plot SED of sample galaxies
plot_sed(combined_fields, centerwave_out)
```

```{code-cell} ipython3
# Interpolation of new galaxies based on previous data points 
plot_prediction(combined_fields, centerwave_out)
```

This interpolation seems to fit equaly fine but it took ~ 5 minutes for training. It is robust and resistant to vast data points compared to Polynomial interpolation because of the ML model used
