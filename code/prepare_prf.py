#function to make the PRF the same pixel scale as the mosaic images and normalized
#tractor expects this, and will give bad results if the pixel scale or normalization is anything else
from astropy.nddata import Cutout2D
import astropy.io.fits as fits
import sys
from reproject import reproject_interp


def prepare_PRF(prf_fitsname, ra_0, dec_0, rotate_angle):
    
    #read in the PRF fits file
    ext_prf =  fits.open(prf_fitsname)[0]

    #fix a type on the header that crashes reproject_interp 
    ext_prf.header['CTYPE1'] = 'RA---TAN'
    
    #ok, need to fake it and make the ra and dec of the center of the prf 
    #be the same as the center of the cutout
    #just using a random cutout here to make this work since we need an image
    #for reproject_interp to work
    ext_prf.header['CRVAL1'] = ra_0
    ext_prf.header['CRVAL2'] = dec_0

    cutout = fits.open('0001_149.96582000_2.53160000_irac_ch1_go2_sci_10.fits')[0]

    prf_resample, footprint = reproject_interp(ext_prf, cutout.header)

    #ugg, ok, and check if it is an odd size
    #tractor crashes if the PRF has an even number of pixels
    if (len(prf_resample.data) % 2) < 1:
        prf_resample=Cutout2D(prf_resample, (9,9), (17,17)) 
    
        #and because cutout2D changes data types
        prf_resample = prf_resample.data
    
    #renormalize the PRF so that the sum of all pixels = 1.0
    #again, tractor gives anomolous results if the PRF is normalized any other way
    prf_resample_norm = prf_resample / prf_resample.sum()

    #looks like a rotation might help
    #still working to figure this out, but setting up to let it happen here
    prf_resample_norm_rotate = rotate(prf_resample_norm, rotate_angle)
       
    return prf_resample_norm_rotate


