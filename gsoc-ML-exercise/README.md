## GSOC2024 ML/DL Starter Problem

This is a simple toy problem meant as a pre-application exercise for the GSOC2024 project [Astronomical data enhancement with DL](https://openastronomy.org/gsoc/gsoc2024/#/projects?project=astronomical_data_enhancement_with_dl). 

**Estimated completion time**: a few hours

Please complete all required tasks and whichever optional task(s) allows you to convey your thought process most easily. The goal is to see what your starting points are, not what you could do with several days of research and optimization.

### Overview

Here we will have a more simplified case of the actual project with no time information. We have measurements of galaxies in different wavelengthts (i.e., broadband filters) in five different fields and the task is to bring them all onto a same wavelength footing. A simple notebook to read the galaxy data in the initial filters is in this repository.


### Instructions

1. Clone this repo and checkout the branch gsco-ML-exercise.
2. Write code.
    - _Required_: A simple way to combine all five fields in optical and NIR filters and output one file in the requested wavelengths.
    - _Optional_: Use ML or DL to do this combination.
    - _Optional_: Use prior information in ML by grouping galaxies at similar redshifts/stellarmasses/ etc. which are columns in the initial catalogs.
3. _Required_: Write text. 300 words max, included as a '.md' file.
    - Explain what you did and how you approached the problem.
    - For any code that you did not write but would if you had more time, write down what you would do.
4. _Required_: Open a PR with your code and writeup to merge to the main branch. Have GSoC2024 in the title of your PR.
