## Herschel PACS & SPIRE (from ESA TAP)
from astroquery.esa.hsa import HSA
from astroquery.exceptions import LoginError
from requests.exceptions import ChunkedEncodingError
import tarfile 
from astropy.io import fits
import glob
import pandas as pd
import astropy.constants as const
from astropy import units as u
import os

from data_structures_spec import MultiIndexDFObject

def find_max_flux_column(df):
  """
  Analyzes a DataFrame with flux columns and returns the column with the largest sum.

  Args:
      df (pandas.DataFrame): The DataFrame containing columns with "flux" in the name.

  Returns:
      str: The name of the column with the largest sum of values containing "flux".
  """

  # Filter column names containing "flux"
  flux_cols = [col for col in df.columns if "flux" in col.lower()]

  # Check if any flux columns are found
  if not flux_cols:
    raise ValueError("No columns containing 'flux' found in the DataFrame")

  # Calculate the sum of each flux column
  flux_sums = {col: df[col].sum() for col in flux_cols}

  # Find the column with the largest sum
  max_flux_col = max(flux_sums, key=flux_sums.get)

  return max_flux_col


def Herschel_get_spec(sample_table, search_radius_arcsec, datadir, delete_tarfiles = False):
    '''
    Retrieves Herschel spectra from a subset of modes for a list of sources.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    search_radius_arcsec : `float`
        Search radius in arcseconds.
    datadir : `str`
        Data directory where to store the data. Each function will create a
        separate data directory (for example "[datadir]/HST/" for HST data).
    delete_tarfiles: True/False
        Should the tarfiles be deteled after spectra are extracted?
        
    Returns
    -------
    df_spec : MultiIndexDFObject
        The main data structure to store all spectra
        
    '''

    ## Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:
        search_coords = stab["coord"]
        print("working on object", stab["label"])
        
        #first find the object ids from herschel then download the data for each observation id
        #query_hsa_tap doesn't accept an upload_table, so do this so do this as a for loop over each instrument and object..
 
        for instrument_name in ['PACS', 'SPIRE']:
            querystring = "select top 10 observation_id from hsa.v_active_observation join hsa.instrument using (instrument_oid) where contains(point('ICRS', hsa.v_active_observation.ra, hsa.v_active_observation.dec), circle('ICRS', "+str(search_coords.ra.deg)+", " + str(search_coords.dec.deg) +", " + str(search_radius_arcsec) +"))=1 and hsa.instrument.instrument_name='"+str(instrument_name)+"'"
            objectid_table = HSA.query_hsa_tap(querystring)

            #download_data only accepts one observation_id so we need to loop over each observation_id
            for tab_id in range(len(objectid_table)):
                observation_id = str(objectid_table[tab_id]['observation_id'])
                try: 
                    HSA.download_data(observation_id=observation_id, retrieval_type='OBSERVATION', 
                            instrument_name=instrument_name, product_level = "LEVEL2, LEVEL_2_5, LEVEL_3", download_dir = datadir)
                
                    #ok, now we have the tar files, need to read those into the right data structure
                    #first untar
                    path_to_file = f"{datadir}/{observation_id}.tar"
                    
                    object = tarfile.open(path_to_file, 'r')
                    #there are a million files!!! how do I know which one I need?
                    #only grab the files which have the final spectra in them = "HPSSPEC" in directory name
                    #not all modes have a final spectrum (cubes?) 
                    for member in object.getmembers():
                         if "HPSSPEC"  in member.name: 
                            path_to_final_dir = f'data/herschel/final_spectrum{observation_id}'
                            object.extract(member, path = path_to_final_dir)

                            for directory_name in os.listdir(path_to_final_dir):
 
                                for fits_file_path in glob.glob(f"{path_to_final_dir}/{directory_name}/{observation_id}/level*/HPSSPEC*/herschel*/*"):
                                    #open the fits file
                                    hdulist = fits.open(fits_file_path) 
                                
                                    #convert final spectrum to pandas dataframe
                                    df = pd.DataFrame(hdulist[1].data)
                                    #There are multiple flux columns; figure out which flux column to use
                                    #advice from https://www.cosmos.esa.int/documents/12133/996891/Product+decision+trees 
                                    #is to use the flux coluimn with the most flux
                                    max_flux = find_max_flux_column(df)
                                    #use the corresponding uncertainty column
                                    max_error =  max_flux.replace("Flux", "Error")

                                    #convert to cgs units for saving and plotting
                                    flux_Jy = df[max_flux].to_numpy() * u.Jy
                                    wavelength = df.wave[0] * u.micrometer #single wavelength for conversion to cgs
                                    flux_cgs = flux_Jy.to(u.erg / u.second / (u.centimeter**2) / u.hertz) * (const.c.to(u.angstrom/u.second)) / (wavelength.to(u.angstrom)**2)
                                    flux_cgs = flux_cgs.to(u.erg / u.second / (u.centimeter**2) / u.angstrom)

                                    flux_err_Jy = df[max_error].to_numpy() * u.Jy
                                    flux_err_cgs = flux_err_Jy.to(u.erg / u.second / (u.centimeter**2) / u.hertz) * (const.c.to(u.angstrom/u.second)) / (wavelength.to(u.angstrom)**2)
                                    flux_err_cgs = flux_err_cgs.to(u.erg / u.second / (u.centimeter**2) / u.angstrom)
                                    
                                    wave = df.wave.to_numpy() * u.micrometer
                                    wave = wave.to(u.angstrom)
                                    #build the df with this object's spectrum from this instrument
                                    dfsingle = pd.DataFrame(dict(wave=[wave] , flux=[flux_cgs], err=[flux_err_cgs],
                                                label=[stab["label"]],
                                                objectid=[stab["objectid"]],
                                                mission=["Herschel"],
                                                instrument=[instrument_name],
                                                filter=[df["band"][0]],
                                                )).set_index(["objectid", "label", "filter", "mission"])
                                    

                                    df_spec.append(dfsingle)

                except LoginError:
                    print("This observation is proprietary, which might mean that it is calibration data")
                except:
                    print("Connection to the ESA archive broken")

                #delete tar files
                if delete_tarfiles:
                    filename_tar = f"data/herschel/{objectid_table[tab_id]['observation_id']}.tar"
                    print('filename_tar', filename_tar)
                    if os.path.exists(filename_tar):
                        print('removing tar file')
                        os.remove(filename_tar)

    return df_spec
                    