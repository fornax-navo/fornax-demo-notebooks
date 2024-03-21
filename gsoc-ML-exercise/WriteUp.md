By transforming the data into Pandas dataframe we can see that each table has around 40000 rows. From column names it is clear that the columns represent features, so the row indeces represent a range of wavelengths in Angstroms at which the features are measured. This aligns well with the filter wavelength values specified above and with the background of the project in general; in this exercise we are basically augmenting the dataset and pcking out specific samples. Hence, to solve the problem we can just interpolate the whole table for each field (doing it column by column), I used 1D linear interpolation. Then what is left to do is to simply pick out the desired frequencies from the dataframe. Below is an example plot of how unaugmented feature compares to an augmented one. The feature present is maximum flux from F105 Galaxy. (shift along y axis was added for better visuals)

![image.png](https://github.com/VladZenko/GSoC_ML_2024/blob/06b6f5cb6a4259e11ea7e5879c677c6fceb22abf/plots/ReadCandels_35_0.png)

The process is then repeated for each field (expressed as a dataframe) and  saved as FITS file, where each field is saved as a separate HDU.

To even more reflect on the topic of the project, I made an attempt to implement NN trained on defined values to predict the undefined ones. For simplicity and to save time, I took one of the columns with undefined values (-99) and made a simple multi-layer perceptron for regression problem. I augmented the data by adding random perturbations within 10% of the value to increase dataset size. Then I normalised the data (min-max and log) because the samples were low in orders of magnitude. Below is the result of training.


![image-2.png](https://github.com/VladZenko/GSoC_ML_2024/blob/06b6f5cb6a4259e11ea7e5879c677c6fceb22abf/plots/ReadCandels_55_1.png)

If I spent more time with the code I would have tried different normalisation methods (power law and Yeo-Johnson) to achieve more accurate fit of the data with NNs. Also, I would have tried applying LSTMs for predictions since the data is sequential (flux measurements vary with and, hence, are a function of wavelength).
