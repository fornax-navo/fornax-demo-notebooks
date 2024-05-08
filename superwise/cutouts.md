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

By the IPAC Science Platform Team, started: Apr 24, 2024- last edit: May 1st, 2024

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


gs = fits.getdata('../../../Dropbox/WFIRST_WPS/CANDELS_fits/gds.fits')
sel1 = (gs['zbest']>0.01)&(gs['zbest']<0.1)&(gs['CLASS_STAR']<0.95)&(gs['Hmag']<24)&(gs['FWHM_IMAGE']>15)
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
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('code_src/')

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
for boz in range(5):
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
    plt.show()
```

## Training a unet model

```{code-cell} ipython3
import torch
import torch.nn as nn

class ConditionalDiffusionModel(nn.Module):
    def __init__(self):
        super(ConditionalDiffusionModel, self).__init__()
        # Starting with an initial convolution to manage the input
        self.initial_conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        # Encoder and Decoder with explicit size management
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(64, 64))  # Ensuring the output is upscaled directly to 64x64
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # Output to match the HR size
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, dataloader, epochs, device):
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            low_res = inputs[:, 0, 18:-18, 18:-18].unsqueeze(1).to(device)  # Prepare low-res input: WISE W1
            high_res = inputs[:, 3, :, :].unsqueeze(1).to(device)  # Prepare high-res targets: IRAC Ch1
            
            optimizer.zero_grad()
            outputs = model(low_res)
            loss = criterion(outputs, high_res)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

```{code-cell} ipython3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConditionalDiffusionModel().to(device)
train(model, dataloader, epochs=150, device=device)
torch.save(model, 'model_complete.pth')
```

## Using the trained model:

```{code-cell} ipython3
model = torch.load('model_complete.pth')
model.eval()
```

```{code-cell} ipython3
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5,shuffle=True, num_workers=int(0))
inputs, classes = next(iter(dataloader))  

for k in range(inputs.shape[0]):
    w1in = inputs[k, 0, 18:-18, 18:-18].cpu().numpy()  
    ch1in = inputs[k, 3, :,:].cpu().numpy()  

    cutout = inputs[k, 0, 18:-18, 18:-18].unsqueeze(0).unsqueeze(0).to(device)  # Process and move in one step
    with torch.no_grad():  # Inference without tracking gradients
        output = model(cutout)


    plt.figure(figsize=(8,3))
    plt.subplot(1,3,1)
    plt.imshow(w1in,origin='lower')
    plt.text(1,1,'W1',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,3,2)
    output_image = output.squeeze().cpu().numpy()  # Remove batch and channel dims and convert to numpy
    plt.imshow(output_image,origin='lower')  # Assuming grayscale output
    plt.text(1,1,'W1 Enhanced',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(ch1in,origin='lower')
    plt.text(1,1,'IRAC Ch1',fontsize=10,color='y')
    plt.axis('off')
    plt.show()
```

# Two channel unet model

```{code-cell} ipython3
import torch
import torch.nn as nn

class ConditionalDiffusionModel(nn.Module):
    def __init__(self):
        super(ConditionalDiffusionModel, self).__init__()
        # Encoder: upscales and increases depth
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)  # Initial upsampling
        )
        # Additional layers to correctly shape the output
        self.processing = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(64, 64)),  # Ensure exact output dimensions
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Decoder: reduces depth back down to 2 channels
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)  # Output 2 channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.processing(x)
        x = self.decoder(x)
        return x

def train(model, dataloader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0]  # Adjust based on the structure of your batch
            # Assume classes or other data might be in batch[1], batch[2], etc.
            
            # Extract and prepare low-res and high-res images
            low_res = inputs[:, 0:2, 18:-18, 18:-18].to(device)
            high_res = inputs[:, 2:4, :, :].to(device)
            
            optimizer.zero_grad()
            outputs = model(low_res)
            loss = criterion(outputs, high_res)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConditionalDiffusionModel().to(device)
train(model, dataloader, epochs=50, device=device)
torch.save(model, 'model_complete_2band.pth')
```

```{code-cell} ipython3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConditionalDiffusionModel().to(device)
train(model, dataloader, epochs=50, device=device)
torch.save(model, 'model_complete_2band.pth')
```

```{code-cell} ipython3
model = torch.load('model_complete_2band.pth')
model.eval()
```

```{code-cell} ipython3
device='cpu'
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=int(0))
inputs, classes = next(iter(dataloader))  

for k in range(inputs.shape[0]):
    w1in = inputs[k, 0, 18:-18, 18:-18].cpu().numpy()  
    w2in = inputs[k, 1, 18:-18, 18:-18].cpu().numpy()  
    ch1in = inputs[k, 2, :,:].cpu().numpy()  
    ch2in = inputs[k, 3, :,:].cpu().numpy()  
    

    cutout = inputs[k, 0:2, 18:-18, 18:-18].unsqueeze(0).to(device)  # Process and move in one step
    with torch.no_grad():  # Inference without tracking gradients
        output = model(cutout)


    plt.figure(figsize=(12,3))
    plt.subplot(1,6,1)
    plt.imshow(w1in,origin='lower')
    plt.text(1,1,'W1',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,2)
    plt.imshow(w2in,origin='lower')
    plt.text(1,1,'W2',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,3)
    output_image = output.squeeze().cpu().numpy()  
    plt.imshow(output_image[0,:,:],origin='lower')  
    plt.text(1,2,'W1 DL-Enhanced',fontsize=10,color='y')
    plt.axis('off')


    plt.subplot(1,6,4)
    plt.imshow(output_image[1,:,:],origin='lower')  
    plt.text(1,2,'W2 DL-Enhanced',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,5)
    plt.imshow(ch1in,origin='lower')
    plt.text(1,1,'IRAC Ch1',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,6)
    plt.imshow(ch2in,origin='lower')
    plt.text(1,1,'IRAC Ch2',fontsize=10,color='y')
    plt.axis('off')

    plt.tight_layout()
    #plt.show()
    
    plt.savefig('test.png')
    
    
    
    
    
```

# Some diffusion

```{code-cell} ipython3
import torch
import torch.nn as nn

class SuperResolutionDiffusionModel(nn.Module):
    def __init__(self):
        super(SuperResolutionDiffusionModel, self).__init__()
        # Encoder: upscales and increases depth
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)  # Initial upsampling
        )
        # Processing layers to ensure correct output dimensions
        self.processing = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(64, 64)),  # Upscale to target size
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Decoder: reduces depth back down to 2 channels
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)  # Output 2 channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.processing(x)
        x = self.decoder(x)
        return x

def train(model, dataloader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0].to(device)
            low_res = inputs[:, 0:2, :, :]  # Prepare low-res input
            high_res = inputs[:, 2:4, :, :].to(device)  # Prepare high-res targets
            
            # Add noise for the diffusion simulation
            noise = torch.randn_like(high_res) * 0.1  # Noise level can be adjusted
            noisy_high_res = high_res + noise
            
            optimizer.zero_grad()
            outputs = model(low_res)
            loss = criterion(outputs, noisy_high_res)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuperResolutionDiffusionModel().to(device)
# Assuming dataloader and epochs are defined elsewhere
train(model, dataloader, 50, device)
torch.save(model.state_dict(), 'model_complete_2band.pth')
```

```{code-cell} ipython3
model = SuperResolutionDiffusionModel().to(device)
model.load_state_dict(torch.load('model_complete_2band.pth'))
model.eval()
```

```{code-cell} ipython3
device='cpu'
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=int(0))
inputs, classes = next(iter(dataloader))  

for k in range(inputs.shape[0]):
    w1in = inputs[k, 0, 18:-18, 18:-18].cpu().numpy()  
    w2in = inputs[k, 1, 18:-18, 18:-18].cpu().numpy()  
    ch1in = inputs[k, 2, :,:].cpu().numpy()  
    ch2in = inputs[k, 3, :,:].cpu().numpy()  
    

    cutout = inputs[k, 0:2, 18:-18, 18:-18].unsqueeze(0).to(device)  # Process and move in one step
    with torch.no_grad():  # Inference without tracking gradients
        output = model(cutout)


    plt.figure(figsize=(12,3))
    plt.subplot(1,6,1)
    plt.imshow(w1in,origin='lower')
    plt.text(1,1,'W1',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,2)
    plt.imshow(w2in,origin='lower')
    plt.text(1,1,'W2',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,3)
    output_image = output.squeeze().cpu().numpy()  
    plt.imshow(output_image[0,:,:],origin='lower')  
    plt.text(1,2,'W1 DL-Enhanced',fontsize=10,color='y')
    plt.axis('off')


    plt.subplot(1,6,4)
    plt.imshow(output_image[1,:,:],origin='lower')  
    plt.text(1,2,'W2 DL-Enhanced',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,5)
    plt.imshow(ch1in,origin='lower')
    plt.text(1,1,'IRAC Ch1',fontsize=10,color='y')
    plt.axis('off')

    plt.subplot(1,6,6)
    plt.imshow(ch2in,origin='lower')
    plt.text(1,1,'IRAC Ch2',fontsize=10,color='y')
    plt.axis('off')

    plt.tight_layout()
    #plt.show()
    
    #plt.savefig('test.png')
    
    
    
    
    
```

```{code-cell} ipython3

```
