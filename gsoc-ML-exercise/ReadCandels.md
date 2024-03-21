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

# How to read the data


```python
import astropy.io.fits as fits
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def abmag(flux):
    return np.where(flux > 0, -2.5 * np.log10(flux) + 23.9, -99.0)

```

## Reading in the input files from Box


```python
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


```python
## Reading in the 2nd field: GOODS-N 
url = 'https://caltech.box.com/shared/static/jv3gyp0kkxnbql5wnpodujjn4cvchrud'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    gn = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download GOODS-N.")
    
centerwave_gn = {'KPNO_U_FLUX':3647,'ACS_F435W_FLUX':4317,'ACS_F606W_FLUX':5918,'ACS_F775W_FLUX':7617,'ACS_F814W_FLUX':8047,'ACS_F850LP_FLUX':9055,'WFC3_F105W_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'CFHT_KS_FLUX':21460,'IRAC_CH1_SCANDELS_FLUX':36000}

```


```python
## Reading in the 3rd field: UDS 
url = 'https://caltech.box.com/shared/static/q8oxrb3zisw0xnekrocuydxwoivge91x'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    uds = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download UDS.")
    

centerwave_uds = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'SUBARU_r_FLUX':5960,'ACS_F606W_FLUX':6500,'SUBARU_i_FLUX':7680,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8890,'HAWKI_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'HAWKI_KS_FLUX':21470,'IRAC_CH1_SEDS_FLUX':36000}

```


```python
## Reading in the 4th field: EGS 
url = 'https://caltech.box.com/shared/static/sthjm6vl6b8bdhvg38lyursps9xnoc6h'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    egs = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download EGS.")
    
centerwave_egs = {'CFHT_U_FLUX':3825,'CFHT_G_FLUX':4810,'ACS_F606W_FLUX':5960,'CFHT_R_FLUX':6250,'CFHT_I_FLUX':7690,'ACS_F814W_FLUX':8090,'CFHT_Z_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'WIRCAM_K_FLUX':21460,'IRAC_CH1_FLUX':36000}

```


```python
## Reading in the 5th field: COSMOS
url = 'https://caltech.box.com/shared/static/27xwurf6t3yj2i4mn0atk7gof4y22ow9'
response = requests.get(url, allow_redirects=True)
if response.status_code == 200:
    cos = fits.getdata(BytesIO(response.content))
else:
    print("Failed to download COSMOS.")
    
centerwave_cos = {'CFHT_U_FLUX':3825,'SUBARU_B_FLUX':4500,'CFHT_G_FLUX':5960,'ACS_F606W_FLUX':6500,'CFHT_I_FLUX':7619,'ACS_F814W_FLUX':8047,'SUBARU_Z_FLUX':8829,'ULTRAVISTA_Y_FLUX':10215,'WFC3_F125W_FLUX':12536,'WFC3_F160W_FLUX':15370,'ULTRAVISTA_Ks_FLUX':21521,'IRAC_Ch1_FLUX':36000}

```

## Plotting the SED (spectral energy distribution) of one data point 
Flux as a function of wavelength


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




    Text(0, 0.5, 'Flux (microJansky)')




    
![png](https://github.com/VladZenko/GSoC_ML_2024/blob/4bdd69a7276394ff27da46debc652a3d8c652683/plots/ReadCandels_9_1.png)
    


# We need output to be a catalog combined of all 5 fields with flux in the filters below:


```python
centerwave_out = {'u':3826,'g':4877,'r':6231,'i':7618,'z':8828,'y':10214,'j':12535,'h':14453,'k':17316}
```

---

## Combining the data.


```python
idx = 255
ja = 0
jb = 0
counter = 1

plt.figure(figsize=(12,6))
plt.title('SED of a galaxy in the GS ; GN fields')
for w in centerwave_gs:
    
    if gs[w][idx] > 0:
        plt.plot(centerwave_gs[w],np.log(gs[w][idx])-1,'r*',markersize=10) # add a shift of 1 along the flux axis for visualisational purposes (looks bit nicer)
        if ja == counter:
            plt.axvline(x=centerwave_gs[w], color='r', linestyle='--', linewidth=2, label=f'original $\lambda$= {centerwave_gs[w]} $\AA$ (GS)')
        ja += 1

for w in centerwave_gn:

    if gn[w][idx] > 0: 
        plt.plot(centerwave_gn[w],np.log(gn[w][idx]),'g*',markersize=10)
        if jb == counter:
            plt.axvline(x=centerwave_gn[w], color='g', linestyle='--', linewidth=2, label=f'original $\lambda$= {centerwave_gn[w]} $\AA$ (GN)')     
        jb += 1  

for w in centerwave_out.values():
    plt.axvline(x=w, color='k', linestyle='--', linewidth=0.5, label=f'output $\lambda$= {w} $\AA$')


#plt.yscale('log')
plt.xlabel(f'Wavelength($\AA$)')
plt.ylabel(f'Flux ($mJy$) ')
#plt.xlim(2500, 20000)
plt.legend()
plt.show()

```


    
![png](https://github.com/VladZenko/GSoC_ML_2024/blob/4bdd69a7276394ff27da46debc652a3d8c652683/plots/ReadCandels_14_0.png)
    


### Points are not aligned to the desired wavelength range, which is the cornerstone of the problem.

# 1) Interpolation

### Obvious first approach is to interpolate the initial wavelengths to the given new values based on the $\AA$ separation between original and desired wavebands.


```python
import pandas as pd

print(pd.DataFrame(gs).shape[0])
print(pd.DataFrame(gn).shape[0])
print(pd.DataFrame(uds).shape[0])
print(pd.DataFrame(egs).shape[0])
print(pd.DataFrame(cos).shape[0])
```

    34930
    35445
    35932
    41457
    38671
    


```python
import pandas as pd

df = pd.DataFrame(gs)
df.filter(like='FLUX').head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FLUX_MAX_F435W</th>
      <th>FLUX_MAX_F606W</th>
      <th>FLUX_MAX_F775W</th>
      <th>FLUX_MAX_F814W</th>
      <th>FLUX_MAX_F850LP</th>
      <th>FLUX_MAX_F098M</th>
      <th>FLUX_MAX_F105W</th>
      <th>FLUX_MAX_F125W</th>
      <th>FLUX_MAX_F160W</th>
      <th>FLUX_ISO_F435W</th>
      <th>...</th>
      <th>IRAC_CH3_FLUXERR</th>
      <th>IRAC_CH4_FLUX</th>
      <th>IRAC_CH4_FLUXERR</th>
      <th>FLUX_ISO</th>
      <th>FLUXERR_ISO</th>
      <th>FLUX_AUTO</th>
      <th>FLUXERR_AUTO</th>
      <th>FLUX_RADIUS_1</th>
      <th>FLUX_RADIUS_2</th>
      <th>FLUX_RADIUS_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.003969</td>
      <td>0.034969</td>
      <td>0.111974</td>
      <td>0.121883</td>
      <td>0.167291</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.383018</td>
      <td>0.619589</td>
      <td>1.572560</td>
      <td>...</td>
      <td>0.591554</td>
      <td>35.969700</td>
      <td>0.523036</td>
      <td>54.839500</td>
      <td>0.116213</td>
      <td>54.836200</td>
      <td>0.119412</td>
      <td>3.231</td>
      <td>9.632</td>
      <td>24.208</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-99.000000</td>
      <td>0.002727</td>
      <td>-99.000000</td>
      <td>0.002556</td>
      <td>0.004857</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.014582</td>
      <td>0.012035</td>
      <td>-99.000000</td>
      <td>...</td>
      <td>0.706397</td>
      <td>1.406680</td>
      <td>0.541094</td>
      <td>0.109848</td>
      <td>0.029626</td>
      <td>0.376055</td>
      <td>0.225026</td>
      <td>1.760</td>
      <td>2.679</td>
      <td>3.526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-99.000000</td>
      <td>0.026005</td>
      <td>-99.000000</td>
      <td>0.041599</td>
      <td>0.046668</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.074737</td>
      <td>0.087944</td>
      <td>-99.000000</td>
      <td>...</td>
      <td>0.721565</td>
      <td>7.874050</td>
      <td>0.578550</td>
      <td>8.901660</td>
      <td>0.093645</td>
      <td>10.550200</td>
      <td>0.258220</td>
      <td>3.225</td>
      <td>6.674</td>
      <td>12.288</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-99.000000</td>
      <td>0.002313</td>
      <td>-99.000000</td>
      <td>0.006726</td>
      <td>0.005673</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.006430</td>
      <td>0.005929</td>
      <td>-99.000000</td>
      <td>...</td>
      <td>0.699897</td>
      <td>-0.325973</td>
      <td>0.557147</td>
      <td>0.080305</td>
      <td>0.019593</td>
      <td>0.881169</td>
      <td>3.590050</td>
      <td>3.080</td>
      <td>6.212</td>
      <td>15.002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.001428</td>
      <td>0.003307</td>
      <td>0.003882</td>
      <td>0.002633</td>
      <td>0.003256</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.003887</td>
      <td>0.007169</td>
      <td>0.048994</td>
      <td>...</td>
      <td>0.606722</td>
      <td>2.526120</td>
      <td>0.527717</td>
      <td>0.245176</td>
      <td>0.017457</td>
      <td>0.308574</td>
      <td>0.028042</td>
      <td>1.808</td>
      <td>3.391</td>
      <td>5.283</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-99.000000</td>
      <td>0.010548</td>
      <td>-99.000000</td>
      <td>0.012062</td>
      <td>0.012131</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.015020</td>
      <td>0.013660</td>
      <td>-99.000000</td>
      <td>...</td>
      <td>0.645725</td>
      <td>0.479455</td>
      <td>0.521698</td>
      <td>0.874996</td>
      <td>0.040737</td>
      <td>1.374060</td>
      <td>0.079889</td>
      <td>3.073</td>
      <td>6.311</td>
      <td>11.725</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.003035</td>
      <td>0.006853</td>
      <td>0.011805</td>
      <td>0.011245</td>
      <td>0.012810</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.025250</td>
      <td>0.038633</td>
      <td>0.624920</td>
      <td>...</td>
      <td>0.758402</td>
      <td>10.766300</td>
      <td>0.658811</td>
      <td>7.952390</td>
      <td>0.113713</td>
      <td>9.554740</td>
      <td>0.213527</td>
      <td>5.126</td>
      <td>11.845</td>
      <td>21.625</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.002382</td>
      <td>0.002904</td>
      <td>0.004112</td>
      <td>0.003722</td>
      <td>0.004796</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.004918</td>
      <td>0.004357</td>
      <td>0.020010</td>
      <td>...</td>
      <td>0.725899</td>
      <td>3.633580</td>
      <td>0.649447</td>
      <td>0.052957</td>
      <td>0.008415</td>
      <td>0.146131</td>
      <td>0.024359</td>
      <td>1.713</td>
      <td>2.975</td>
      <td>4.325</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.003430</td>
      <td>0.006522</td>
      <td>0.011728</td>
      <td>0.008975</td>
      <td>0.011088</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.013681</td>
      <td>0.010071</td>
      <td>0.072114</td>
      <td>...</td>
      <td>0.611055</td>
      <td>0.259089</td>
      <td>0.526380</td>
      <td>0.337459</td>
      <td>0.022012</td>
      <td>0.545227</td>
      <td>0.049504</td>
      <td>2.090</td>
      <td>4.128</td>
      <td>10.637</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.001557</td>
      <td>0.002598</td>
      <td>0.005380</td>
      <td>0.005012</td>
      <td>0.005346</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.009016</td>
      <td>0.007275</td>
      <td>0.121200</td>
      <td>...</td>
      <td>0.606722</td>
      <td>4.981120</td>
      <td>0.521029</td>
      <td>0.413750</td>
      <td>0.025110</td>
      <td>0.755778</td>
      <td>0.058069</td>
      <td>3.177</td>
      <td>6.839</td>
      <td>11.701</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000530</td>
      <td>0.000489</td>
      <td>0.001127</td>
      <td>0.003053</td>
      <td>0.001421</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.000000</td>
      <td>0.003233</td>
      <td>0.002900</td>
      <td>...</td>
      <td>0.565551</td>
      <td>1.718580</td>
      <td>0.492938</td>
      <td>0.045009</td>
      <td>0.009456</td>
      <td>0.524286</td>
      <td>0.062834</td>
      <td>3.620</td>
      <td>5.131</td>
      <td>6.358</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.002361</td>
      <td>0.004679</td>
      <td>0.007603</td>
      <td>0.006925</td>
      <td>0.007264</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.010189</td>
      <td>0.009908</td>
      <td>0.077628</td>
      <td>...</td>
      <td>0.647892</td>
      <td>0.940605</td>
      <td>0.575874</td>
      <td>0.399658</td>
      <td>0.035327</td>
      <td>1.666090</td>
      <td>0.162748</td>
      <td>4.011</td>
      <td>7.308</td>
      <td>11.970</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000704</td>
      <td>0.002050</td>
      <td>0.005085</td>
      <td>0.004117</td>
      <td>0.005953</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.009125</td>
      <td>0.008594</td>
      <td>0.017299</td>
      <td>...</td>
      <td>0.576385</td>
      <td>-0.500577</td>
      <td>0.502302</td>
      <td>0.448140</td>
      <td>0.024162</td>
      <td>0.787532</td>
      <td>0.059834</td>
      <td>2.841</td>
      <td>6.080</td>
      <td>12.302</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-99.000000</td>
      <td>0.005187</td>
      <td>-99.000000</td>
      <td>0.010278</td>
      <td>0.012422</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.014660</td>
      <td>0.020199</td>
      <td>-99.000000</td>
      <td>...</td>
      <td>0.647892</td>
      <td>0.597031</td>
      <td>0.537750</td>
      <td>0.898260</td>
      <td>0.038178</td>
      <td>1.177910</td>
      <td>0.076214</td>
      <td>2.243</td>
      <td>4.593</td>
      <td>8.602</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.000458</td>
      <td>-99.000000</td>
      <td>-99.000000</td>
      <td>0.000816</td>
      <td>-99.000000</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.000487</td>
      <td>0.001513</td>
      <td>0.000607</td>
      <td>...</td>
      <td>0.517880</td>
      <td>-0.016193</td>
      <td>0.462171</td>
      <td>0.019671</td>
      <td>0.007544</td>
      <td>0.176628</td>
      <td>0.022853</td>
      <td>2.804</td>
      <td>4.199</td>
      <td>5.719</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-99.000000</td>
      <td>0.002561</td>
      <td>-99.000000</td>
      <td>0.002514</td>
      <td>0.002724</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.004356</td>
      <td>0.009707</td>
      <td>-99.000000</td>
      <td>...</td>
      <td>0.637058</td>
      <td>1.022270</td>
      <td>0.534406</td>
      <td>0.156695</td>
      <td>0.017339</td>
      <td>0.227301</td>
      <td>0.028907</td>
      <td>1.340</td>
      <td>2.507</td>
      <td>4.167</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.000886</td>
      <td>0.001305</td>
      <td>0.002616</td>
      <td>0.002309</td>
      <td>0.003157</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.002861</td>
      <td>0.003694</td>
      <td>0.022229</td>
      <td>...</td>
      <td>0.552550</td>
      <td>-0.230223</td>
      <td>0.474879</td>
      <td>0.131109</td>
      <td>0.012527</td>
      <td>0.339756</td>
      <td>0.037178</td>
      <td>3.022</td>
      <td>6.126</td>
      <td>11.384</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.001242</td>
      <td>0.003443</td>
      <td>0.006210</td>
      <td>0.004373</td>
      <td>0.005508</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.008708</td>
      <td>0.014317</td>
      <td>0.100325</td>
      <td>...</td>
      <td>0.600221</td>
      <td>1.225040</td>
      <td>0.517685</td>
      <td>1.010750</td>
      <td>0.041551</td>
      <td>1.305380</td>
      <td>0.072562</td>
      <td>2.940</td>
      <td>5.317</td>
      <td>9.192</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.002263</td>
      <td>0.003363</td>
      <td>0.005099</td>
      <td>0.005100</td>
      <td>0.006941</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.009914</td>
      <td>0.007076</td>
      <td>0.013595</td>
      <td>...</td>
      <td>0.524381</td>
      <td>1.339800</td>
      <td>0.464846</td>
      <td>0.071985</td>
      <td>0.012847</td>
      <td>0.271502</td>
      <td>0.045649</td>
      <td>1.992</td>
      <td>3.728</td>
      <td>6.038</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.009999</td>
      <td>0.021631</td>
      <td>0.025052</td>
      <td>0.024475</td>
      <td>0.026328</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>0.027324</td>
      <td>0.026960</td>
      <td>1.095050</td>
      <td>...</td>
      <td>0.576385</td>
      <td>1.225740</td>
      <td>0.504308</td>
      <td>3.210630</td>
      <td>0.065520</td>
      <td>3.833680</td>
      <td>0.103160</td>
      <td>3.483</td>
      <td>7.032</td>
      <td>12.644</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 365 columns</p>
</div>




```python
print("Number of rows:", len(df))
print("Shape of DataFrame:", df.shape)
```

    Number of rows: 34930
    Shape of DataFrame: (34930, 717)
    


```python
flux_array = np.transpose(df.filter(like='FLUX').to_numpy())
np.shape(flux_array)
```




    (365, 34930)



### From observing and making sense of the data we can see that the tables cover a range of wavelengths from 0 to around 36000 Angstroms. For each wavelength a value of a certain physical parameter is given (not everything in the tables represents flux). This makes a problem easy, we can just interpolate the whole table, column by column, and then pick out the desired wavelength values fom the provided dictionary. So the solution of the problem is generation of combined and augmented mini-dataset of observed flux from a range of galaxies (9 of them if I am not wrong)


```python
import scipy

flux_array_copy = np.copy(flux_array)


for i in range(np.shape(flux_array)[0]):

    defined_indices = np.where(flux_array_copy[i] != -99)[0]
    undefined_indices = np.where(flux_array_copy[i] == -99)[0]

    defined_values = flux_array_copy[i][defined_indices]

    interp_func = scipy.interpolate.interp1d(defined_indices, defined_values, kind='linear', fill_value="extrapolate")

    flux_array_copy[i][undefined_indices] = interp_func(undefined_indices)
```


```python
print(-99 in flux_array[0])
print(-99 in flux_array_copy[0])
```

    True
    False
    


```python
new_df = pd.DataFrame(np.transpose(flux_array_copy), columns=list(df.filter(like='FLUX').columns)).loc[list(centerwave_out.values())]
new_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FLUX_MAX_F435W</th>
      <th>FLUX_MAX_F606W</th>
      <th>FLUX_MAX_F775W</th>
      <th>FLUX_MAX_F814W</th>
      <th>FLUX_MAX_F850LP</th>
      <th>FLUX_MAX_F098M</th>
      <th>FLUX_MAX_F105W</th>
      <th>FLUX_MAX_F125W</th>
      <th>FLUX_MAX_F160W</th>
      <th>FLUX_ISO_F435W</th>
      <th>...</th>
      <th>IRAC_CH3_FLUXERR</th>
      <th>IRAC_CH4_FLUX</th>
      <th>IRAC_CH4_FLUXERR</th>
      <th>FLUX_ISO</th>
      <th>FLUXERR_ISO</th>
      <th>FLUX_AUTO</th>
      <th>FLUXERR_AUTO</th>
      <th>FLUX_RADIUS_1</th>
      <th>FLUX_RADIUS_2</th>
      <th>FLUX_RADIUS_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3826</th>
      <td>0.001267</td>
      <td>0.006885</td>
      <td>0.028717</td>
      <td>0.032234</td>
      <td>0.046362</td>
      <td>3.815595</td>
      <td>0.075319</td>
      <td>0.109809</td>
      <td>0.189851</td>
      <td>0.470517</td>
      <td>...</td>
      <td>0.370534</td>
      <td>25.207700</td>
      <td>0.371877</td>
      <td>25.982300</td>
      <td>0.061242</td>
      <td>26.062500</td>
      <td>0.063224</td>
      <td>3.822</td>
      <td>8.675</td>
      <td>16.435</td>
    </tr>
    <tr>
      <th>4877</th>
      <td>0.001407</td>
      <td>0.002282</td>
      <td>0.005997</td>
      <td>0.005251</td>
      <td>0.006407</td>
      <td>3.490182</td>
      <td>0.010484</td>
      <td>0.010683</td>
      <td>0.014748</td>
      <td>0.023647</td>
      <td>...</td>
      <td>0.589387</td>
      <td>6.798190</td>
      <td>0.672758</td>
      <td>0.557846</td>
      <td>0.012986</td>
      <td>0.636032</td>
      <td>0.021305</td>
      <td>1.867</td>
      <td>3.795</td>
      <td>6.865</td>
    </tr>
    <tr>
      <th>6231</th>
      <td>0.002503</td>
      <td>0.002662</td>
      <td>0.003212</td>
      <td>0.002360</td>
      <td>0.002747</td>
      <td>3.070953</td>
      <td>0.004734</td>
      <td>0.004939</td>
      <td>0.004620</td>
      <td>0.069856</td>
      <td>...</td>
      <td>602.355000</td>
      <td>1.260980</td>
      <td>0.471280</td>
      <td>0.150715</td>
      <td>0.010845</td>
      <td>0.211093</td>
      <td>0.021362</td>
      <td>1.940</td>
      <td>3.632</td>
      <td>6.124</td>
    </tr>
    <tr>
      <th>7618</th>
      <td>0.001249</td>
      <td>0.003331</td>
      <td>0.013019</td>
      <td>0.013150</td>
      <td>0.024027</td>
      <td>2.641507</td>
      <td>0.007357</td>
      <td>0.073356</td>
      <td>0.112241</td>
      <td>0.026721</td>
      <td>...</td>
      <td>0.544276</td>
      <td>5.796420</td>
      <td>0.577284</td>
      <td>4.799710</td>
      <td>0.045880</td>
      <td>4.981090</td>
      <td>0.052877</td>
      <td>1.958</td>
      <td>4.132</td>
      <td>8.161</td>
    </tr>
    <tr>
      <th>8828</th>
      <td>0.001754</td>
      <td>0.002874</td>
      <td>0.004693</td>
      <td>0.003935</td>
      <td>0.004198</td>
      <td>2.266863</td>
      <td>0.004143</td>
      <td>0.004133</td>
      <td>0.004661</td>
      <td>0.081998</td>
      <td>...</td>
      <td>0.425377</td>
      <td>0.365950</td>
      <td>0.465264</td>
      <td>0.358075</td>
      <td>0.012168</td>
      <td>0.490451</td>
      <td>0.027931</td>
      <td>4.109</td>
      <td>6.481</td>
      <td>12.247</td>
    </tr>
    <tr>
      <th>10214</th>
      <td>0.007353</td>
      <td>0.017607</td>
      <td>0.024067</td>
      <td>0.022957</td>
      <td>0.025179</td>
      <td>1.837727</td>
      <td>0.026392</td>
      <td>0.028331</td>
      <td>0.031807</td>
      <td>1.487960</td>
      <td>...</td>
      <td>0.374082</td>
      <td>2.528190</td>
      <td>0.394502</td>
      <td>5.269860</td>
      <td>0.029492</td>
      <td>5.907860</td>
      <td>0.038189</td>
      <td>4.587</td>
      <td>10.647</td>
      <td>19.832</td>
    </tr>
    <tr>
      <th>12535</th>
      <td>0.000061</td>
      <td>0.000081</td>
      <td>0.000236</td>
      <td>0.000419</td>
      <td>0.000272</td>
      <td>1.119093</td>
      <td>0.000345</td>
      <td>0.000393</td>
      <td>0.000501</td>
      <td>-0.000097</td>
      <td>...</td>
      <td>0.335061</td>
      <td>0.352933</td>
      <td>0.358814</td>
      <td>0.016174</td>
      <td>0.001611</td>
      <td>0.030339</td>
      <td>0.005319</td>
      <td>2.243</td>
      <td>4.122</td>
      <td>6.895</td>
    </tr>
    <tr>
      <th>14453</th>
      <td>0.000271</td>
      <td>0.000359</td>
      <td>0.000360</td>
      <td>0.000214</td>
      <td>0.000264</td>
      <td>0.525237</td>
      <td>0.000442</td>
      <td>0.000721</td>
      <td>0.000884</td>
      <td>0.003682</td>
      <td>...</td>
      <td>0.271572</td>
      <td>0.250919</td>
      <td>0.286257</td>
      <td>0.016395</td>
      <td>0.001354</td>
      <td>0.018030</td>
      <td>0.002610</td>
      <td>1.258</td>
      <td>2.267</td>
      <td>3.441</td>
    </tr>
    <tr>
      <th>17316</th>
      <td>0.000229</td>
      <td>0.000464</td>
      <td>0.000593</td>
      <td>0.000830</td>
      <td>0.001170</td>
      <td>0.001367</td>
      <td>0.001889</td>
      <td>0.001375</td>
      <td>0.001723</td>
      <td>0.000046</td>
      <td>...</td>
      <td>0.288470</td>
      <td>-0.390232</td>
      <td>0.334353</td>
      <td>0.023736</td>
      <td>0.003418</td>
      <td>0.055353</td>
      <td>0.007587</td>
      <td>1.759</td>
      <td>2.883</td>
      <td>4.200</td>
    </tr>
  </tbody>
</table>
<p>9 rows × 365 columns</p>
</div>



### The <u>following cell contains the solution</u>: a function that takes in the initial data files and combines them into a single augmented dataset of 5 separate tables, and exports it as FITS file.

---


```python
from astropy.table import Table, vstack

data_files = [gs, gn, uds, egs, cos]



def combine_datasets(data_list, cw_out_dict, save_path):

    combined_data = []
    columns = []
    hdus = [fits.PrimaryHDU()]

    for i in range(len(data_list)):

        df = pd.DataFrame(data_list[i])

        columns.append(list(df.filter(like='FLUX').columns))

        flux_array = np.transpose(df.filter(like='FLUX').to_numpy()) # only select columns representing flux-associated quantities

        flux_array_copy = np.copy(flux_array)

        for j in range(np.shape(flux_array)[0]):

            defined_indices = np.where(flux_array_copy[j] != -99)[0]
            undefined_indices = np.where(flux_array_copy[j] == -99)[0]

            defined_values = flux_array_copy[j][defined_indices]

            interp_func = scipy.interpolate.interp1d(defined_indices, defined_values, kind='linear', fill_value="extrapolate")

            flux_array_copy[j][undefined_indices] = interp_func(undefined_indices)


        combined_data.append(np.transpose(flux_array_copy)[list(cw_out_dict.values())])

    #---------------------------------------------------------------------
    # Save as FITS file:
        table = Table(np.transpose(flux_array_copy)[list(cw_out_dict.values())], names=columns[i])
        hdu = fits.BinTableHDU(table)
        hdus.append(hdu)  # one HDU for each table


    hdulist = fits.HDUList(hdus)
    hdulist.writeto(save_path, overwrite=True)

    print('FITS file saved, containing data for all fields at the specified wavelengths. combined data and column names also returned locally.')

    return combined_data, columns

```


```python
cd, clmns = combine_datasets(data_files, centerwave_out, r"C:\Users\vovaz\Desktop\Combined_Data.fits")
```

    FITS file saved, containing data for all fields at the specified wavelengths. combined data and column names also returned locally.
    

---


```python
hdulist = fits.open(r"C:\Users\vovaz\Desktop\Combined_Data.fits")
```


```python
len(hdulist) # Initial HDU always empty, 1+5=6
```




    6




```python
data = hdulist[1].data
df = pd.DataFrame(data)
df.index = list(centerwave_out.values())
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FLUX_MAX_F435W</th>
      <th>FLUX_MAX_F606W</th>
      <th>FLUX_MAX_F775W</th>
      <th>FLUX_MAX_F814W</th>
      <th>FLUX_MAX_F850LP</th>
      <th>FLUX_MAX_F098M</th>
      <th>FLUX_MAX_F105W</th>
      <th>FLUX_MAX_F125W</th>
      <th>FLUX_MAX_F160W</th>
      <th>FLUX_ISO_F435W</th>
      <th>...</th>
      <th>IRAC_CH3_FLUXERR</th>
      <th>IRAC_CH4_FLUX</th>
      <th>IRAC_CH4_FLUXERR</th>
      <th>FLUX_ISO</th>
      <th>FLUXERR_ISO</th>
      <th>FLUX_AUTO</th>
      <th>FLUXERR_AUTO</th>
      <th>FLUX_RADIUS_1</th>
      <th>FLUX_RADIUS_2</th>
      <th>FLUX_RADIUS_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3826</th>
      <td>0.001267</td>
      <td>0.006885</td>
      <td>0.028717</td>
      <td>0.032234</td>
      <td>0.046362</td>
      <td>3.815595</td>
      <td>0.075319</td>
      <td>0.109809</td>
      <td>0.189851</td>
      <td>0.470517</td>
      <td>...</td>
      <td>0.370534</td>
      <td>25.207700</td>
      <td>0.371877</td>
      <td>25.982300</td>
      <td>0.061242</td>
      <td>26.062500</td>
      <td>0.063224</td>
      <td>3.822</td>
      <td>8.675</td>
      <td>16.435</td>
    </tr>
    <tr>
      <th>4877</th>
      <td>0.001407</td>
      <td>0.002282</td>
      <td>0.005997</td>
      <td>0.005251</td>
      <td>0.006407</td>
      <td>3.490182</td>
      <td>0.010484</td>
      <td>0.010683</td>
      <td>0.014748</td>
      <td>0.023647</td>
      <td>...</td>
      <td>0.589387</td>
      <td>6.798190</td>
      <td>0.672758</td>
      <td>0.557846</td>
      <td>0.012986</td>
      <td>0.636032</td>
      <td>0.021305</td>
      <td>1.867</td>
      <td>3.795</td>
      <td>6.865</td>
    </tr>
    <tr>
      <th>6231</th>
      <td>0.002503</td>
      <td>0.002662</td>
      <td>0.003212</td>
      <td>0.002360</td>
      <td>0.002747</td>
      <td>3.070953</td>
      <td>0.004734</td>
      <td>0.004939</td>
      <td>0.004620</td>
      <td>0.069856</td>
      <td>...</td>
      <td>602.355000</td>
      <td>1.260980</td>
      <td>0.471280</td>
      <td>0.150715</td>
      <td>0.010845</td>
      <td>0.211093</td>
      <td>0.021362</td>
      <td>1.940</td>
      <td>3.632</td>
      <td>6.124</td>
    </tr>
    <tr>
      <th>7618</th>
      <td>0.001249</td>
      <td>0.003331</td>
      <td>0.013019</td>
      <td>0.013150</td>
      <td>0.024027</td>
      <td>2.641507</td>
      <td>0.007357</td>
      <td>0.073356</td>
      <td>0.112241</td>
      <td>0.026721</td>
      <td>...</td>
      <td>0.544276</td>
      <td>5.796420</td>
      <td>0.577284</td>
      <td>4.799710</td>
      <td>0.045880</td>
      <td>4.981090</td>
      <td>0.052877</td>
      <td>1.958</td>
      <td>4.132</td>
      <td>8.161</td>
    </tr>
    <tr>
      <th>8828</th>
      <td>0.001754</td>
      <td>0.002874</td>
      <td>0.004693</td>
      <td>0.003935</td>
      <td>0.004198</td>
      <td>2.266863</td>
      <td>0.004143</td>
      <td>0.004133</td>
      <td>0.004661</td>
      <td>0.081998</td>
      <td>...</td>
      <td>0.425377</td>
      <td>0.365950</td>
      <td>0.465264</td>
      <td>0.358075</td>
      <td>0.012168</td>
      <td>0.490451</td>
      <td>0.027931</td>
      <td>4.109</td>
      <td>6.481</td>
      <td>12.247</td>
    </tr>
    <tr>
      <th>10214</th>
      <td>0.007353</td>
      <td>0.017607</td>
      <td>0.024067</td>
      <td>0.022957</td>
      <td>0.025179</td>
      <td>1.837727</td>
      <td>0.026392</td>
      <td>0.028331</td>
      <td>0.031807</td>
      <td>1.487960</td>
      <td>...</td>
      <td>0.374082</td>
      <td>2.528190</td>
      <td>0.394502</td>
      <td>5.269860</td>
      <td>0.029492</td>
      <td>5.907860</td>
      <td>0.038189</td>
      <td>4.587</td>
      <td>10.647</td>
      <td>19.832</td>
    </tr>
    <tr>
      <th>12535</th>
      <td>0.000061</td>
      <td>0.000081</td>
      <td>0.000236</td>
      <td>0.000419</td>
      <td>0.000272</td>
      <td>1.119093</td>
      <td>0.000345</td>
      <td>0.000393</td>
      <td>0.000501</td>
      <td>-0.000097</td>
      <td>...</td>
      <td>0.335061</td>
      <td>0.352933</td>
      <td>0.358814</td>
      <td>0.016174</td>
      <td>0.001611</td>
      <td>0.030339</td>
      <td>0.005319</td>
      <td>2.243</td>
      <td>4.122</td>
      <td>6.895</td>
    </tr>
    <tr>
      <th>14453</th>
      <td>0.000271</td>
      <td>0.000359</td>
      <td>0.000360</td>
      <td>0.000214</td>
      <td>0.000264</td>
      <td>0.525237</td>
      <td>0.000442</td>
      <td>0.000721</td>
      <td>0.000884</td>
      <td>0.003682</td>
      <td>...</td>
      <td>0.271572</td>
      <td>0.250919</td>
      <td>0.286257</td>
      <td>0.016395</td>
      <td>0.001354</td>
      <td>0.018030</td>
      <td>0.002610</td>
      <td>1.258</td>
      <td>2.267</td>
      <td>3.441</td>
    </tr>
    <tr>
      <th>17316</th>
      <td>0.000229</td>
      <td>0.000464</td>
      <td>0.000593</td>
      <td>0.000830</td>
      <td>0.001170</td>
      <td>0.001367</td>
      <td>0.001889</td>
      <td>0.001375</td>
      <td>0.001723</td>
      <td>0.000046</td>
      <td>...</td>
      <td>0.288470</td>
      <td>-0.390232</td>
      <td>0.334353</td>
      <td>0.023736</td>
      <td>0.003418</td>
      <td>0.055353</td>
      <td>0.007587</td>
      <td>1.759</td>
      <td>2.883</td>
      <td>4.200</td>
    </tr>
  </tbody>
</table>
<p>9 rows × 365 columns</p>
</div>




```python
df1 = pd.DataFrame(gs)
df1 = df1.filter(like='FLUX').loc[list(centerwave_out.values())]
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FLUX_MAX_F435W</th>
      <th>FLUX_MAX_F606W</th>
      <th>FLUX_MAX_F775W</th>
      <th>FLUX_MAX_F814W</th>
      <th>FLUX_MAX_F850LP</th>
      <th>FLUX_MAX_F098M</th>
      <th>FLUX_MAX_F105W</th>
      <th>FLUX_MAX_F125W</th>
      <th>FLUX_MAX_F160W</th>
      <th>FLUX_ISO_F435W</th>
      <th>...</th>
      <th>IRAC_CH3_FLUXERR</th>
      <th>IRAC_CH4_FLUX</th>
      <th>IRAC_CH4_FLUXERR</th>
      <th>FLUX_ISO</th>
      <th>FLUXERR_ISO</th>
      <th>FLUX_AUTO</th>
      <th>FLUXERR_AUTO</th>
      <th>FLUX_RADIUS_1</th>
      <th>FLUX_RADIUS_2</th>
      <th>FLUX_RADIUS_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3826</th>
      <td>0.001267</td>
      <td>0.006885</td>
      <td>0.028717</td>
      <td>0.032234</td>
      <td>0.046362</td>
      <td>-99.0</td>
      <td>0.075319</td>
      <td>0.109809</td>
      <td>0.189851</td>
      <td>0.470517</td>
      <td>...</td>
      <td>0.370534</td>
      <td>25.207700</td>
      <td>0.371877</td>
      <td>25.982300</td>
      <td>0.061242</td>
      <td>26.062500</td>
      <td>0.063224</td>
      <td>3.822</td>
      <td>8.675</td>
      <td>16.435</td>
    </tr>
    <tr>
      <th>4877</th>
      <td>0.001407</td>
      <td>0.002282</td>
      <td>0.005997</td>
      <td>0.005251</td>
      <td>0.006407</td>
      <td>-99.0</td>
      <td>-99.000000</td>
      <td>0.010683</td>
      <td>0.014748</td>
      <td>0.023647</td>
      <td>...</td>
      <td>0.589387</td>
      <td>6.798190</td>
      <td>0.672758</td>
      <td>0.557846</td>
      <td>0.012986</td>
      <td>0.636032</td>
      <td>0.021305</td>
      <td>1.867</td>
      <td>3.795</td>
      <td>6.865</td>
    </tr>
    <tr>
      <th>6231</th>
      <td>0.002503</td>
      <td>0.002662</td>
      <td>0.003212</td>
      <td>0.002360</td>
      <td>0.002747</td>
      <td>-99.0</td>
      <td>-99.000000</td>
      <td>0.004939</td>
      <td>0.004620</td>
      <td>0.069856</td>
      <td>...</td>
      <td>602.355000</td>
      <td>1.260980</td>
      <td>0.471280</td>
      <td>0.150715</td>
      <td>0.010845</td>
      <td>0.211093</td>
      <td>0.021362</td>
      <td>1.940</td>
      <td>3.632</td>
      <td>6.124</td>
    </tr>
    <tr>
      <th>7618</th>
      <td>0.001249</td>
      <td>0.003331</td>
      <td>0.013019</td>
      <td>0.013150</td>
      <td>0.024027</td>
      <td>-99.0</td>
      <td>-99.000000</td>
      <td>0.073356</td>
      <td>0.112241</td>
      <td>0.026721</td>
      <td>...</td>
      <td>0.544276</td>
      <td>5.796420</td>
      <td>0.577284</td>
      <td>4.799710</td>
      <td>0.045880</td>
      <td>4.981090</td>
      <td>0.052877</td>
      <td>1.958</td>
      <td>4.132</td>
      <td>8.161</td>
    </tr>
    <tr>
      <th>8828</th>
      <td>0.001754</td>
      <td>0.002874</td>
      <td>0.004693</td>
      <td>0.003935</td>
      <td>0.004198</td>
      <td>-99.0</td>
      <td>0.004143</td>
      <td>0.004133</td>
      <td>0.004661</td>
      <td>0.081998</td>
      <td>...</td>
      <td>0.425377</td>
      <td>0.365950</td>
      <td>0.465264</td>
      <td>0.358075</td>
      <td>0.012168</td>
      <td>0.490451</td>
      <td>0.027931</td>
      <td>4.109</td>
      <td>6.481</td>
      <td>12.247</td>
    </tr>
    <tr>
      <th>10214</th>
      <td>0.007353</td>
      <td>0.017607</td>
      <td>0.024067</td>
      <td>0.022957</td>
      <td>0.025179</td>
      <td>-99.0</td>
      <td>0.026392</td>
      <td>0.028331</td>
      <td>0.031807</td>
      <td>1.487960</td>
      <td>...</td>
      <td>0.374082</td>
      <td>2.528190</td>
      <td>0.394502</td>
      <td>5.269860</td>
      <td>0.029492</td>
      <td>5.907860</td>
      <td>0.038189</td>
      <td>4.587</td>
      <td>10.647</td>
      <td>19.832</td>
    </tr>
    <tr>
      <th>12535</th>
      <td>0.000061</td>
      <td>0.000081</td>
      <td>0.000236</td>
      <td>0.000419</td>
      <td>0.000272</td>
      <td>-99.0</td>
      <td>0.000345</td>
      <td>0.000393</td>
      <td>0.000501</td>
      <td>-0.000097</td>
      <td>...</td>
      <td>0.335061</td>
      <td>0.352933</td>
      <td>0.358814</td>
      <td>0.016174</td>
      <td>0.001611</td>
      <td>0.030339</td>
      <td>0.005319</td>
      <td>2.243</td>
      <td>4.122</td>
      <td>6.895</td>
    </tr>
    <tr>
      <th>14453</th>
      <td>0.000271</td>
      <td>0.000359</td>
      <td>0.000360</td>
      <td>0.000214</td>
      <td>0.000264</td>
      <td>-99.0</td>
      <td>0.000442</td>
      <td>0.000721</td>
      <td>0.000884</td>
      <td>0.003682</td>
      <td>...</td>
      <td>0.271572</td>
      <td>0.250919</td>
      <td>0.286257</td>
      <td>0.016395</td>
      <td>0.001354</td>
      <td>0.018030</td>
      <td>0.002610</td>
      <td>1.258</td>
      <td>2.267</td>
      <td>3.441</td>
    </tr>
    <tr>
      <th>17316</th>
      <td>0.000229</td>
      <td>0.000464</td>
      <td>0.000593</td>
      <td>0.000830</td>
      <td>0.001170</td>
      <td>-99.0</td>
      <td>0.001889</td>
      <td>0.001375</td>
      <td>0.001723</td>
      <td>0.000046</td>
      <td>...</td>
      <td>0.288470</td>
      <td>-0.390232</td>
      <td>0.334353</td>
      <td>0.023736</td>
      <td>0.003418</td>
      <td>0.055353</td>
      <td>0.007587</td>
      <td>1.759</td>
      <td>2.883</td>
      <td>4.200</td>
    </tr>
  </tbody>
</table>
<p>9 rows × 365 columns</p>
</div>




```python


plt.figure(figsize=(13,7))
plt.title('SED of the galaxy F105W in the GS field')
plt.subplot(111)

asd = []
for w in list(centerwave_out.values()):
    if df1['FLUX_MAX_F105W'][w] > 0:
        asd.append(w)
        plt.plot(w, np.log(df1['FLUX_MAX_F105W'][w]),'r*-',markersize=10)
    

    plt.plot(w, np.log(df['FLUX_MAX_F105W'][w])-1,'g*',markersize=10)

plt.plot(asd, np.log(list(df1[df1['FLUX_MAX_F105W']>0]['FLUX_MAX_F105W'])), 'r--', linewidth=0.5, label='Original')
plt.plot(list(centerwave_out.values()), np.log(df['FLUX_MAX_F105W'])-1, 'g--', linewidth=0.5, label='Augmented')

for w in centerwave_out.values():
    plt.axvline(x=w, color='k', linestyle='--', linewidth=0.5, label=f'output $\lambda$= {w} $\AA$')



#plt.yscale('log')
plt.xlabel(f'Wavelength($\AA$)')
plt.ylabel(f'Flux ($mJy$) log scaled')
#plt.xlim(2500, 20000)
plt.legend()
plt.show()
```


    
![png](https://github.com/VladZenko/GSoC_ML_2024/blob/4bdd69a7276394ff27da46debc652a3d8c652683/plots/ReadCandels_35_0.png)
    


___

### Instead of using scipy we can be fancy and make a simple regressor NN to predict the values for use. This will bring the mock task close to the main project. Since the task is already solved, I restrain myself to one column only. We need to extract the values that are defined (not equal to -99) to serve as labels and corresponding wavelengths to act as the input training data. Additionally, augmentation might be worth looking at since one label per one training sample is not very promising for a NN. Can introduce augmentation by taking small deviations from the ground truth (small relative to the dataset, probably within 1 sigma of the STD).


```python
df_gs = pd.DataFrame(gs)
df_gs = df_gs.filter(like='FLUX_MAX_F435W')
df_gs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FLUX_MAX_F435W</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.003969</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-99.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-99.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-99.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.001428</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>34925</th>
      <td>-99.000000</td>
    </tr>
    <tr>
      <th>34926</th>
      <td>0.000364</td>
    </tr>
    <tr>
      <th>34927</th>
      <td>0.000617</td>
    </tr>
    <tr>
      <th>34928</th>
      <td>0.000459</td>
    </tr>
    <tr>
      <th>34929</th>
      <td>0.000315</td>
    </tr>
  </tbody>
</table>
<p>34930 rows × 1 columns</p>
</div>




```python
x_data = list(df_gs[df_gs['FLUX_MAX_F435W']>0].index)
y_data = list(df_gs[df_gs['FLUX_MAX_F435W']>0]['FLUX_MAX_F435W'])
```


```python
np.std(y_data)
```




    0.21920860747039295



### Distribution too flat, cannot use std for augmentation. Instead, use +-10% of values to augment the data


```python
x_data_aug = np.repeat(x_data,5)

y_data_aug = np.repeat(y_data,5)

random_percentages = np.random.uniform(-0.1, 0.1, size=len(y_data_aug))
random_values = y_data_aug * random_percentages
random_values[4::5] = 0 #make sure to keep the original value in the dataset as well


y_data_aug = y_data_aug + random_values

y_max = np.max(y_data_aug)
y_min = np.min(y_data_aug)

print(y_min, y_max)

# normalize y data (0-1) to simplify the job for our very simplistic NN
y_data_norm = (y_data_aug - y_min)/(y_max - y_min)
print(np.min(y_data_norm),  np.max(y_data_norm))

# make a function to denormalise predictions when done training
def denormalize_prediction(norm_prediction, min_val, max_val):

    denormalized_prediction = norm_prediction*(max_val - min_val) + min_val
    return denormalized_prediction
```

    3.0300931639964885e-07 14.626799487966098
    0.0 1.0
    


```python
len(y_data_norm)
```




    166360




```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_data_aug, y_data_norm, test_size=0.15, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


model = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(1,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')


callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, verbose=1, mode='min', min_lr=1e-6),
             EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)]
```


```python
history = model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=callbacks)


```

    Epoch 1/100
    

    4419/4419 [==============================] - 11s 2ms/step - loss: 223.6600 - val_loss: 2.8377e-04 - lr: 0.0010
    Epoch 2/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 454.4906 - val_loss: 4.3836e-04 - lr: 0.0010
    Epoch 3/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.0294 - val_loss: 4.5610e-04 - lr: 0.0010
    Epoch 4/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 4.2006 - val_loss: 3.7117e-04 - lr: 0.0010
    Epoch 5/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.6269 - val_loss: 1.9099e-04 - lr: 0.0010
    Epoch 6/100
    4403/4419 [============================>.] - ETA: 0s - loss: 0.0068
    Epoch 6: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0067 - val_loss: 0.0012 - lr: 0.0010
    Epoch 7/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0026 - val_loss: 5.2755e-04 - lr: 2.5000e-04
    Epoch 8/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0114 - val_loss: 1.9465e-04 - lr: 2.5000e-04
    Epoch 9/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0052 - val_loss: 4.7062e-04 - lr: 2.5000e-04
    Epoch 10/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0259 - val_loss: 2.3742e-04 - lr: 2.5000e-04
    Epoch 11/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0071 - val_loss: 1.8067e-04 - lr: 2.5000e-04
    Epoch 12/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0020 - val_loss: 1.9103e-04 - lr: 2.5000e-04
    Epoch 13/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0039 - val_loss: 2.1495e-04 - lr: 2.5000e-04
    Epoch 14/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0079 - val_loss: 1.7643e-04 - lr: 2.5000e-04
    Epoch 15/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0012 - val_loss: 0.0014 - lr: 2.5000e-04
    Epoch 16/100
    4409/4419 [============================>.] - ETA: 0s - loss: 0.0011
    Epoch 16: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
    4419/4419 [==============================] - 10s 2ms/step - loss: 0.0011 - val_loss: 7.5729e-04 - lr: 2.5000e-04
    Epoch 17/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 3.1248e-04 - val_loss: 1.7606e-04 - lr: 6.2500e-05
    Epoch 18/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 3.4352e-04 - val_loss: 1.7847e-04 - lr: 6.2500e-05
    Epoch 19/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 3.1707e-04 - val_loss: 1.8117e-04 - lr: 6.2500e-05
    Epoch 20/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 3.1483e-04 - val_loss: 1.7623e-04 - lr: 6.2500e-05
    Epoch 21/100
    4411/4419 [============================>.] - ETA: 0s - loss: 2.6355e-04
    Epoch 21: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.6321e-04 - val_loss: 1.7725e-04 - lr: 6.2500e-05
    Epoch 22/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3921e-04 - val_loss: 1.7579e-04 - lr: 1.5625e-05
    Epoch 23/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.4107e-04 - val_loss: 1.7583e-04 - lr: 1.5625e-05
    Epoch 24/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.4010e-04 - val_loss: 1.7586e-04 - lr: 1.5625e-05
    Epoch 25/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3792e-04 - val_loss: 1.7581e-04 - lr: 1.5625e-05
    Epoch 26/100
    4399/4419 [============================>.] - ETA: 0s - loss: 2.3884e-04
    Epoch 26: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3792e-04 - val_loss: 1.7562e-04 - lr: 1.5625e-05
    Epoch 27/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3548e-04 - val_loss: 1.7561e-04 - lr: 3.9063e-06
    Epoch 28/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3549e-04 - val_loss: 1.7564e-04 - lr: 3.9063e-06
    Epoch 29/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3551e-04 - val_loss: 1.7560e-04 - lr: 3.9063e-06
    Epoch 30/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3547e-04 - val_loss: 1.7560e-04 - lr: 3.9063e-06
    Epoch 31/100
    4412/4419 [============================>.] - ETA: 0s - loss: 2.3573e-04
    Epoch 31: ReduceLROnPlateau reducing learning rate to 1e-06.
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3549e-04 - val_loss: 1.7560e-04 - lr: 3.9063e-06
    Epoch 32/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7562e-04 - lr: 1.0000e-06
    Epoch 33/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3529e-04 - val_loss: 1.7562e-04 - lr: 1.0000e-06
    Epoch 34/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7566e-04 - lr: 1.0000e-06
    Epoch 35/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 36/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3529e-04 - val_loss: 1.7563e-04 - lr: 1.0000e-06
    Epoch 37/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3529e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 38/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 39/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 40/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7564e-04 - lr: 1.0000e-06
    Epoch 41/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 42/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7562e-04 - lr: 1.0000e-06
    Epoch 43/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7568e-04 - lr: 1.0000e-06
    Epoch 44/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 45/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7562e-04 - lr: 1.0000e-06
    Epoch 46/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7562e-04 - lr: 1.0000e-06
    Epoch 47/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7565e-04 - lr: 1.0000e-06
    Epoch 48/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3527e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 49/100
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7565e-04 - lr: 1.0000e-06
    Epoch 50/100
    4400/4419 [============================>.] - ETA: 0s - loss: 2.3612e-04Restoring model weights from the end of the best epoch: 30.
    4419/4419 [==============================] - 10s 2ms/step - loss: 2.3528e-04 - val_loss: 1.7561e-04 - lr: 1.0000e-06
    Epoch 50: early stopping
    


```python
type(x_data_aug[0])
```




    numpy.int32




```python
predictions = model.predict(np.array([3826, 4877, 6231, 7618, 8828, 10214, 12535, 14453, 17316], dtype=np.int32))
predictions_denorm = denormalize_prediction(predictions, y_min, y_max)
predictions_denorm
```

    1/1 [==============================] - 0s 98ms/step
    




    array([[0.01462015],
           [0.01412103],
           [0.01347871],
           [0.0128194 ],
           [0.0122453 ],
           [0.01158729],
           [0.0104853 ],
           [0.00957468],
           [0.0082142 ]], dtype=float32)




```python
data = hdulist[1].data
df = pd.DataFrame(data)
df.index = list(centerwave_out.values())
df['FLUX_MAX_F435W']
```




    3826     0.001267
    4877     0.001407
    6231     0.002503
    7618     0.001249
    8828     0.001754
    10214    0.007353
    12535    0.000061
    14453    0.000271
    17316    0.000229
    Name: FLUX_MAX_F435W, dtype: float64



### The results are not reliable, they are one order of magnitude off from the ground truth. This is likely a result of the original data containing very small values clustered in a relatively wide range, so that even normalized values are still <<1. This can potentially be solved with log normalisation.


```python
y_data_norm_log = np.log(y_data_aug) # add 1 to account for 0 if present
print(np.min(y_data_norm_log),  np.max(y_data_norm_log))

# make a function to denormalise predictions when done training
def denormalize_prediction_log(norm_prediction, min_val, max_val):

    denormalized_prediction = np.exp(norm_prediction)
    return denormalized_prediction
```

    -15.009502284716167 2.6828554274610967
    


```python
# For DL we ideally want the skew in the data being close to 0:

from scipy.stats import skew

original_skew = skew(y_data_aug)
log_skew = skew(y_data_norm_log)
min_max_skew = skew(y_data_norm)

print(f"Original: {original_skew}")
print(f"Log: {log_skew}")
print(f"Min-Max: {min_max_skew}")
```

    Original: 41.34535742822595
    Log: 0.6753140649538087
    Min-Max: 41.34535742822596
    


```python
x_train, x_val, y_train, y_val = train_test_split(x_data_aug, y_data_norm_log, test_size=0.15, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_ds = train_ds.batch(64)
val_ds = val_ds.batch(64)


model1 = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(1,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model1.compile(optimizer='adam', loss='mean_squared_error')


callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, verbose=1, mode='min', min_lr=1e-6)
             ,EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
             ]

history1 = model1.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=callbacks)
```

    Epoch 1/100
     144/2210 [>.............................] - ETA: 4s - loss: 1577.0602

    2210/2210 [==============================] - 6s 2ms/step - loss: 121.4063 - val_loss: 8.6610 - lr: 0.0010
    Epoch 2/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 8.2541 - val_loss: 14.7537 - lr: 0.0010
    Epoch 3/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 6.2466 - val_loss: 22.1055 - lr: 0.0010
    Epoch 4/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 30.3740 - val_loss: 3.6258 - lr: 0.0010
    Epoch 5/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 3.2903 - val_loss: 2.7758 - lr: 0.0010
    Epoch 6/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 2.6053 - val_loss: 2.4852 - lr: 0.0010
    Epoch 7/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 2.8893 - val_loss: 1.4757 - lr: 0.0010
    Epoch 8/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.6001 - val_loss: 1.3340 - lr: 0.0010
    Epoch 9/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.4982 - val_loss: 1.6244 - lr: 0.0010
    Epoch 10/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3485 - val_loss: 1.3147 - lr: 0.0010
    Epoch 11/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3267 - val_loss: 1.3068 - lr: 0.0010
    Epoch 12/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3301 - val_loss: 1.3071 - lr: 0.0010
    Epoch 13/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3247 - val_loss: 1.3178 - lr: 0.0010
    Epoch 14/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3358 - val_loss: 1.3062 - lr: 0.0010
    Epoch 15/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3259 - val_loss: 1.3093 - lr: 0.0010
    Epoch 16/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3219 - val_loss: 1.3153 - lr: 0.0010
    Epoch 17/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3253 - val_loss: 1.3155 - lr: 0.0010
    Epoch 18/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3176 - val_loss: 1.3072 - lr: 0.0010
    Epoch 19/100
    2190/2210 [============================>.] - ETA: 0s - loss: 1.3169
    Epoch 19: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3163 - val_loss: 1.3069 - lr: 0.0010
    Epoch 20/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3096 - val_loss: 1.3074 - lr: 2.5000e-04
    Epoch 21/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3098 - val_loss: 1.3074 - lr: 2.5000e-04
    Epoch 22/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3098 - val_loss: 1.3092 - lr: 2.5000e-04
    Epoch 23/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3095 - val_loss: 1.3071 - lr: 2.5000e-04
    Epoch 24/100
    2204/2210 [============================>.] - ETA: 0s - loss: 1.3096
    Epoch 24: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3095 - val_loss: 1.3070 - lr: 2.5000e-04
    Epoch 25/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3081 - val_loss: 1.3054 - lr: 6.2500e-05
    Epoch 26/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3080 - val_loss: 1.3054 - lr: 6.2500e-05
    Epoch 27/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3081 - val_loss: 1.3054 - lr: 6.2500e-05
    Epoch 28/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.3011 - val_loss: 1.2876 - lr: 6.2500e-05
    Epoch 29/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2856 - val_loss: 1.2778 - lr: 6.2500e-05
    Epoch 30/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2780 - val_loss: 1.2714 - lr: 6.2500e-05
    Epoch 31/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2729 - val_loss: 1.2670 - lr: 6.2500e-05
    Epoch 32/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2690 - val_loss: 1.2643 - lr: 6.2500e-05
    Epoch 33/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2660 - val_loss: 1.2622 - lr: 6.2500e-05
    Epoch 34/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2635 - val_loss: 1.2603 - lr: 6.2500e-05
    Epoch 35/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2615 - val_loss: 1.2580 - lr: 6.2500e-05
    Epoch 36/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2601 - val_loss: 1.2582 - lr: 6.2500e-05
    Epoch 37/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2594 - val_loss: 1.2565 - lr: 6.2500e-05
    Epoch 38/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2576 - val_loss: 1.2558 - lr: 6.2500e-05
    Epoch 39/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2566 - val_loss: 1.2540 - lr: 6.2500e-05
    Epoch 40/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2551 - val_loss: 1.2537 - lr: 6.2500e-05
    Epoch 41/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2544 - val_loss: 1.2519 - lr: 6.2500e-05
    Epoch 42/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2533 - val_loss: 1.2523 - lr: 6.2500e-05
    Epoch 43/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2524 - val_loss: 1.2517 - lr: 6.2500e-05
    Epoch 44/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2516 - val_loss: 1.2511 - lr: 6.2500e-05
    Epoch 45/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2509 - val_loss: 1.2532 - lr: 6.2500e-05
    Epoch 46/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2505 - val_loss: 1.2503 - lr: 6.2500e-05
    Epoch 47/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2497 - val_loss: 1.2501 - lr: 6.2500e-05
    Epoch 48/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2494 - val_loss: 1.2496 - lr: 6.2500e-05
    Epoch 49/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2487 - val_loss: 1.2496 - lr: 6.2500e-05
    Epoch 50/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2486 - val_loss: 1.2485 - lr: 6.2500e-05
    Epoch 51/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2479 - val_loss: 1.2486 - lr: 6.2500e-05
    Epoch 52/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2476 - val_loss: 1.2477 - lr: 6.2500e-05
    Epoch 53/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2470 - val_loss: 1.2478 - lr: 6.2500e-05
    Epoch 54/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2468 - val_loss: 1.2472 - lr: 6.2500e-05
    Epoch 55/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2465 - val_loss: 1.2472 - lr: 6.2500e-05
    Epoch 56/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2460 - val_loss: 1.2469 - lr: 6.2500e-05
    Epoch 57/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2455 - val_loss: 1.2466 - lr: 6.2500e-05
    Epoch 58/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2455 - val_loss: 1.2466 - lr: 6.2500e-05
    Epoch 59/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2457 - val_loss: 1.2466 - lr: 6.2500e-05
    Epoch 60/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2445 - val_loss: 1.2459 - lr: 6.2500e-05
    Epoch 61/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2442 - val_loss: 1.2465 - lr: 6.2500e-05
    Epoch 62/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2436 - val_loss: 1.2453 - lr: 6.2500e-05
    Epoch 63/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2428 - val_loss: 1.2459 - lr: 6.2500e-05
    Epoch 64/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2426 - val_loss: 1.2450 - lr: 6.2500e-05
    Epoch 65/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2425 - val_loss: 1.2458 - lr: 6.2500e-05
    Epoch 66/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2420 - val_loss: 1.2458 - lr: 6.2500e-05
    Epoch 67/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2418 - val_loss: 1.2456 - lr: 6.2500e-05
    Epoch 68/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2414 - val_loss: 1.2435 - lr: 6.2500e-05
    Epoch 69/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2415 - val_loss: 1.2428 - lr: 6.2500e-05
    Epoch 70/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2414 - val_loss: 1.2432 - lr: 6.2500e-05
    Epoch 71/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2413 - val_loss: 1.2419 - lr: 6.2500e-05
    Epoch 72/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2400 - val_loss: 1.2437 - lr: 6.2500e-05
    Epoch 73/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2393 - val_loss: 1.2391 - lr: 6.2500e-05
    Epoch 74/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2383 - val_loss: 1.2411 - lr: 6.2500e-05
    Epoch 75/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2384 - val_loss: 1.2422 - lr: 6.2500e-05
    Epoch 76/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2379 - val_loss: 1.2405 - lr: 6.2500e-05
    Epoch 77/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2374 - val_loss: 1.2385 - lr: 6.2500e-05
    Epoch 78/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2374 - val_loss: 1.2384 - lr: 6.2500e-05
    Epoch 79/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2372 - val_loss: 1.2383 - lr: 6.2500e-05
    Epoch 80/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2368 - val_loss: 1.2376 - lr: 6.2500e-05
    Epoch 81/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2365 - val_loss: 1.2375 - lr: 6.2500e-05
    Epoch 82/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2365 - val_loss: 1.2368 - lr: 6.2500e-05
    Epoch 83/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2364 - val_loss: 1.2372 - lr: 6.2500e-05
    Epoch 84/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2362 - val_loss: 1.2376 - lr: 6.2500e-05
    Epoch 85/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2364 - val_loss: 1.2355 - lr: 6.2500e-05
    Epoch 86/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2359 - val_loss: 1.2355 - lr: 6.2500e-05
    Epoch 87/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2356 - val_loss: 1.2368 - lr: 6.2500e-05
    Epoch 88/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2358 - val_loss: 1.2363 - lr: 6.2500e-05
    Epoch 89/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2355 - val_loss: 1.2359 - lr: 6.2500e-05
    Epoch 90/100
    2188/2210 [============================>.] - ETA: 0s - loss: 1.2363
    Epoch 90: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2356 - val_loss: 1.2355 - lr: 6.2500e-05
    Epoch 91/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2333 - val_loss: 1.2351 - lr: 1.5625e-05
    Epoch 92/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2331 - val_loss: 1.2350 - lr: 1.5625e-05
    Epoch 93/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2330 - val_loss: 1.2349 - lr: 1.5625e-05
    Epoch 94/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2329 - val_loss: 1.2349 - lr: 1.5625e-05
    Epoch 95/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2329 - val_loss: 1.2348 - lr: 1.5625e-05
    Epoch 96/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2329 - val_loss: 1.2348 - lr: 1.5625e-05
    Epoch 97/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2328 - val_loss: 1.2348 - lr: 1.5625e-05
    Epoch 98/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2328 - val_loss: 1.2348 - lr: 1.5625e-05
    Epoch 99/100
    2195/2210 [============================>.] - ETA: 0s - loss: 1.2331
    Epoch 99: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2327 - val_loss: 1.2349 - lr: 1.5625e-05
    Epoch 100/100
    2210/2210 [==============================] - 5s 2ms/step - loss: 1.2323 - val_loss: 1.2339 - lr: 3.9063e-06
    


```python
predictions_log = model1.predict(np.array([3826, 4877, 6231, 7618, 8828, 10214, 12535, 14453, 17316], dtype=np.int32))
predictions_denorm_log = denormalize_prediction_log(predictions_log, y_min, y_max)
predictions_denorm_log
```

    1/1 [==============================] - 0s 61ms/step
    




    array([[0.00190785],
           [0.00179647],
           [0.00168946],
           [0.00158644],
           [0.00150172],
           [0.00141021],
           [0.0012693 ],
           [0.00116908],
           [0.00132722]], dtype=float32)



### This works a bit better. At least the predictions are mostly the same order of magnitude as the ground truth. Perhaps a more complex architecture would be able to solve the problem and bring the values closer to the ground truth. Introduction of LSTMs would be a nice shot.


```python
plt.figure(figsize=(9,6))
plt.title('Maximum Flux value from Galaxy F435W in GS field')
plt.plot([3826, 4877, 6231, 7618, 8828, 10214, 12535, 14453, 17316], df['FLUX_MAX_F435W'], '*r--', markersize=10, label='original predictions')
plt.plot([3826, 4877, 6231, 7618, 8828, 10214, 12535, 14453, 17316], predictions_denorm, '*g--', markersize=10, label='NN data (min-max norm)')
plt.plot([3826, 4877, 6231, 7618, 8828, 10214, 12535, 14453, 17316], predictions_denorm_log, '*b--', markersize=10, label='NN predictions (log norm)')
plt.xlabel(f'Wavelengths, $\AA$')
plt.ylabel(f'Flux, $mJy$')
plt.legend()
```




    <matplotlib.legend.Legend at 0x12fa5137cd0>




    
![png](https://github.com/VladZenko/GSoC_ML_2024/blob/4bdd69a7276394ff27da46debc652a3d8c652683/plots/ReadCandels_55_1.png)
    



```python

```
