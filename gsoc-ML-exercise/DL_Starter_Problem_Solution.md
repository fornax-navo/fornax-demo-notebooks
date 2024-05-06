---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="--xltAUVwbXR" -->
# GSOC2024 ML/DL Starter Problem
Lucas Martin Garcia
<!-- #endregion -->

```python
#!pip install numpy pandas matplotlib astropy scikit-learn tensorflow
```

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
# Solution
<!-- #endregion -->

```python
# Create data list of the galaxy fields
fields = [gs, gn, uds, egs, cos]
centerwave_fields = [centerwave_gs,centerwave_gn,centerwave_uds,centerwave_egs,centerwave_cos]
```

## Linear Interpolation

<!-- #region id="koLNQrnmWW1i" -->
A simple way to combine all five fields in optical and NIR filters would be to interpolate the flux measures with respect to the wavelengths and obtain the flux directly from the interpolated function. The basic interpolation consists of linear interpolation of the flux points of the galaxies.
This Interpolation process of all the galaxies of all fields requires an estimated execution time of 2~3 minutes.
<!-- #endregion -->

```python
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
An now we can plot the as well the original measurements of the same 3 galaxies with the original wavelengths in addition to the interpolated function from which the new values were obtained to see that the interpolation works well:
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

We can clearly see how the interpolated points in the common wavelenghts are obtained directly form the linear interpolation. This is a very simple way to obtain the measurements although there are more advanced interpolation methods that could offer a more reliable solution based on a wider range of information from the measurements we have.


## Polinomial Interpolation

<!-- #region id="koLNQrnmWW1i" -->
Another type of interpolation is Polynomial interpolation. This type of interpolation consists of estimating the unknown values within a range of known data points by fitting a polynomial function, of a degree given, that passes through these points:
<!-- #endregion -->

```python
import warnings

warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned', category=np.RankWarning)

# Function to interpolate flux measures of all galaxies from an specific field in the centerwave chosen
def poly_interpolate__flux(data_field, centerwave_field, centerwave_out, degree = 3):
    interpolate_results = []
    polyn_coeffs = []
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

        # Perform polynomial interpolation
        coeffs = np.polyfit(valid_wavelengths, valid_fluxes, degree)
        interpolated_flux = np.polyval(coeffs, centerwave_out_values)
        interpolate_results.append(interpolated_flux)
        polyn_coeffs.append(coeffs)

    return (np.array(interpolate_results), polyn_coeffs)

# Apply galaxy interpolation to all fields
degree = 4
combined_fields = []
combined_polyn_coeffs = []
for i in range(len(fields)):
    current_interpolated_flux, current_polyn_coeffs  = poly_interpolate__flux(fields[i], centerwave_fields[i], centerwave_out,degree)
    combined_fields.append(current_interpolated_flux)
    combined_polyn_coeffs.append(current_polyn_coeffs)

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
An now we can plot the as well the original measurements of the same 3 galaxies with the original wavelengths in addition to the interpolated function from which the new values were obtained to see that the interpolation works well:
<!-- #endregion -->

```python
num_galaxies = 3
colors = plt.cm.cividis(np.linspace(0, 1, num_galaxies))
colors2 = plt.cm.viridis(np.linspace(0, 1, num_galaxies))
# Plot first five galaxies
plt.figure(figsize=(10, 4))
for i in range(num_galaxies):
    plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], [gs[w][i] for w in centerwave_gs if gs[w][i] > 0],'o',markersize=10,color=colors[i],markeredgecolor='black',markeredgewidth = 0.5)

    # Plot the polynomial fit curve
    wavelength_range = np.linspace(min([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0]), max([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0]), 500)
    poly_flux = np.polyval(combined_polyn_coeffs[0][i], wavelength_range)
    plt.plot(wavelength_range, poly_flux, label=f'Interpolation Galaxy {i+1}', color=colors[i])
    
    # Plot the polynomial interpolation
    plt.plot(list(centerwave_out.values()), combined_fields[i],'*',markersize=10,label=f'Galaxy{i+1}',color=colors2[i],markeredgecolor='black',markeredgewidth = 0.5)

plt.xlabel('Wavelength(A)')
plt.ylabel('Flux Interpolated')
plt.legend(loc='upper left')
plt.title('SED of sample galaxies in the GOODS-S field')
plt.show()
```

The outcomes obtained with polynomial interpolation appear to be more grounded in the information and potential relationships between measurements than with linear interpolation. 
After evaluating polynomial interpolation with varying degrees, it was found that high-degree polynomials led to overfitting, creating a curve with extreme values. On the other hand, polynomials of low degree were unable to accurately model the complexities of the dataset, resulting in a suboptimal fit. Ultimately, a fourth-degree polynomial was determined to be the most suitable choice. This degree effectively balances the risk of overfitting seen in higher-degree polynomials while capturing more data nuances than lower-degree alternatives, offering a well-adjusted interpolation that aligns closely with both the general trend and specific data points.


## SVR Interpolation

<!-- #region id="koLNQrnmWW1i" -->
Now we are going to implement a SVR model to interpolate de measures:
<!-- #endregion -->

```python
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Function to interpolate flux measures of all galaxies from an specific field in the centerwave chosen
def SVR_interpolate_flux(data_field, centerwave_field, centerwave_out):
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
        # Scale data as a Preprocessing step
        scaler = StandardScaler()
        X_train = scaler.fit_transform(valid_wavelengths.reshape(-1, 1))
        X_predict = scaler.transform(np.array(centerwave_out_values).reshape(-1, 1))
        
        # Interpolate the values using the predictions from the SVR model
        svr_model = SVR(kernel='rbf', epsilon=0.1 , C=100, gamma='auto')
        svr_model.fit(X_train, valid_fluxes)
        interpolated_flux = svr_model.predict(X_predict)        
        interpolate_results.append(interpolated_flux)
        
    return np.array(interpolate_results)

# Apply galaxy interpolation to all fields
combined_fields = []
for i in range(len(fields)):
    current_interpolated_flux = SVR_interpolate_flux(fields[i], centerwave_fields[i], centerwave_out)
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
An now we can plot the as well the original measurements of the same 3 galaxies with the original wavelengths in addition to the interpolated function from which the new values were obtained to see that the interpolation works well:
<!-- #endregion -->

```python
num_galaxies = 3
colors = plt.cm.cividis(np.linspace(0, 1, num_galaxies))
colors2 = plt.cm.viridis(np.linspace(0, 1, num_galaxies))
# Plot first five galaxies
plt.figure(figsize=(10, 4))
centerwave_field_values = list(centerwave_fields[0].values())
centerwave_out_values = list(centerwave_out.values())
for i in range(num_galaxies):

    # We need to predict again the range of points to plot the full interpolation
    # Flux measures of the current galaxy, filtering out negative flux values
    current_flux = np.array([gs[filter][i] for filter in centerwave_fields[0]])
    valid_flux_indices = current_flux > 0  # Indices of positive flux values
    if not valid_flux_indices.any():
        # If there are no positive flux values, skip this galaxy or handle accordingly
        continue
    # Only use positive flux values and corresponding wavelengths for interpolation
    valid_fluxes = current_flux[valid_flux_indices]
    valid_wavelengths = np.array(centerwave_field_values)[valid_flux_indices]
    # Interpolate the measures to obtain the flux in the wavelengths specified
    interpolated_flux = np.interp(centerwave_out_values, valid_wavelengths, valid_fluxes)      
    # Scale data as a Preprocessing step
    scaler = StandardScaler()
    X_train = scaler.fit_transform(valid_wavelengths.reshape(-1, 1))
    wavelength_range = np.linspace(min([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0]), max([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0]), 500)
    X_predict = scaler.transform(wavelength_range.reshape(-1, 1))
    # Support Vector Regression (SVR) model
    svr_model = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
    svr_model.fit(X_train, valid_fluxes)
    # Interpolate flux values to obtain flux in requested wavelengths using SVR model
    interpolated_flux = svr_model.predict(X_predict)    
    plt.plot(wavelength_range, interpolated_flux, label=f'Interpolation Galaxy {i+1}', color=colors[i])
    
    #plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], [gs[w][i] for w in centerwave_gs if gs[w][i] > 0], label=f'Interpolation Galaxy {i+1}',color=colors[i])
    plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], [gs[w][i] for w in centerwave_gs if gs[w][i] > 0],'o',markersize=10,color=colors[i],markeredgecolor='black',markeredgewidth = 0.5)
    plt.plot(list(centerwave_out.values()), combined_fields[i],'*',markersize=10,label=f'Galaxy{i+1}',color=colors2[i],markeredgecolor='black',markeredgewidth = 0.5)

plt.xlabel('Wavelength(A)')
plt.ylabel('Flux Interpolated')
plt.legend(loc='upper left')
plt.title('SED of sample galaxies in the GOODS-S field')
plt.show()
```

Using interpolation through a Support Vector Regression (SVR) model demonstrates significantly improved outcomes, indicating that this approach effectively leverages a broader spectrum of information. Unlike simpler interpolation methods, SVR is adept at capturing complex relationships within the data by learning a detailed curve that represents a higher understanding of the actual wavelenght functions. The total computation time with this model is affordable being less than 5 minutes approximately. 
In the implementation of the Support Vector Regression (SVR) model, the parameters were finely tuned after observation of the results. This specific configuration was selected as it yielded the best performance, aligning most closely with the dataset's underlying patterns.


## DNN Interpolation

<!-- #region id="koLNQrnmWW1i" -->
Now we are going to implement a Deep Neural Network model to interpolate de measures:
<!-- #endregion -->

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
# Function to interpolate flux measures of all galaxies from an specific field in the centerwave chosen
def DNN_interpolate_flux(data_field, centerwave_field, centerwave_out):
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
        # Scale data as a Preprocessing step
        scaler = StandardScaler()
        X_train = scaler.fit_transform(valid_wavelengths.reshape(-1, 1))
        X_predict = scaler.transform(np.array(centerwave_out_values).reshape(-1, 1))
        
        # DNN model
        DNN_model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        DNN_model.compile(optimizer=Adam(), loss='mse')
        DNN_model.fit(X_train, valid_fluxes, epochs=1, batch_size=16, verbose=0)
        
        # Interpolate using the DNN model
        interpolated_flux = DNN_model.predict(X_predict, verbose=0) 
        interpolate_results.append(interpolated_flux)
        
    return np.array(interpolate_results)

# For that many galaxies the training time is too high for the scope of this project
# Apply galaxy interpolation to all fields
#combined_fields = []
#for i in range(len(fields)):
    #current_interpolated_flux = DNN_interpolate_flux(fields[i], centerwave_fields[i], centerwave_out)
    #combined_fields.append(current_interpolated_flux)

# Convert to numpy
#combined_fields = np.vstack(combined_fields)
```

<!-- #region id="CnmKHH_hyF7H" -->
As the result, `combined_fields` would be the one file output with all the galaxies in the requested wavelengths.
The computation time to train that many galaxies is too high for the scope of this project evan with a small amount of neurons per layer, but just as an example we can show some galaxies predicted by this model.
<!-- #endregion -->

<!-- #region id="0wE-6lBLxrJ0" -->
We can plot the measurements obtained of the first 3 galaxies in the GOODS-S field with requested wavelengths:
<!-- #endregion -->

```python
num_galaxies = 3
colors = plt.cm.cividis(np.linspace(0, 1, num_galaxies))
colors2 = plt.cm.viridis(np.linspace(0, 1, num_galaxies))
# Plot first five galaxies
plt.figure(figsize=(10, 4))
centerwave_field_values = list(centerwave_fields[0].values())
centerwave_out_values = list(centerwave_out.values())
for i in range(num_galaxies):

    # Flux measures of the current galaxy, filtering out negative flux values
    current_flux = np.array([gs[filter][i] for filter in centerwave_fields[0]])
    valid_flux_indices = current_flux > 0  # Indices of positive flux values
    if not valid_flux_indices.any():
        # If there are no positive flux values, skip this galaxy or handle accordingly
        continue
    # Only use positive flux values and corresponding wavelengths for interpolation
    valid_fluxes = current_flux[valid_flux_indices]
    valid_wavelengths = np.array(centerwave_field_values)[valid_flux_indices]
    # Interpolate the measures to obtain the flux in the wavelengths specified
    interpolated_flux = np.interp(centerwave_out_values, valid_wavelengths, valid_fluxes)      
    # Scale data as a Preprocessing step
    scaler = StandardScaler()
    X_train = scaler.fit_transform(valid_wavelengths.reshape(-1, 1))
    wavelength_range = np.linspace(min([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0]), max([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0]), 500)
    X_predict_function = scaler.transform(wavelength_range.reshape(-1, 1))
    X_predict = scaler.transform(np.array(centerwave_out_values).reshape(-1, 1))
    # DNN model
    DNN_model = Sequential([
        Dense(512, activation='relu', input_shape=(1,)),
        Dense(256, activation='relu', input_shape=(1,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    DNN_model.compile(optimizer=Adam(), loss='mse')
    DNN_model.fit(X_train, valid_fluxes, epochs=110, verbose=0)
    interpolated_function = DNN_model.predict(X_predict_function, verbose=0)    
    interpolated_flux = DNN_model.predict(X_predict, verbose=0)    
    plt.plot(wavelength_range, interpolated_function, label=f'Interpolation Galaxy {i+1}', color=colors[i])
    
    plt.plot([centerwave_gs[w] for w in centerwave_gs if gs[w][i] > 0], [gs[w][i] for w in centerwave_gs if gs[w][i] > 0],'o',markersize=10,color=colors[i],markeredgecolor='black',markeredgewidth = 0.5)
    plt.plot(list(centerwave_out.values()), interpolated_flux,'*',markersize=10,label=f'Galaxy{i+1}',color=colors2[i],markeredgecolor='black',markeredgewidth = 0.5)

plt.xlabel('Wavelength(A)')
plt.ylabel('Flux Interpolated')
plt.legend(loc='upper left')
plt.title('SED of sample galaxies in the GOODS-S field')
plt.show()
```

Using a dense neural network model, the results appear to be less accurately fitted. With extensive training, the model tends to overfit, resembling linear interpolation, while with less training, the outcomes still fail to align well with the values. Additionally, the training demands are significantly high for the scope of this project, presenting practical constraints in terms of time and computational resources. This suggests that while dense neural networks offer powerful modeling capabilities, their application might not be the most efficient or effective choice, at first, for projects with limited resources or when seeking to model data with these specific underlying patterns.


# Final Conclusion


Upon reviewing the outcomes of various interpolation methods, the Support Vector Regression (SVR) model stands out as the most promising. This approach appears to encapsulate a broader array of information from the data measurements, demonstrating superior adaptability and precision in its fit compared to other techniques. Unlike the polynomial interpolation, which required careful balancing between degrees to avoid overfitting or underfitting, with results limited by the polynomial properites, and the dense neural network model, which faced challenges with overfitting and high computational demands, the SVR model effectively captures the complex relationships within the data while maintaining am affordable training.
