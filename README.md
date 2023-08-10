CXO-ICM
===

**Pipeline to analyze Chandra ACIS-I cluster observations**

[![](https://img.shields.io/badge/python-3.*-blue)](https://www.python.org/download/releases/3.0/) [![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

``CXO-ICM`` is a python tool that enables analyzing galaxy cluster observations realized with the ACIS-I instrument on-board Chandra. Besides the usual pre-processing that removes flares and point sources from the event files and that characterizes the background, this software performs a forward modeling of the intracluster medium (ICM) properties such as gas density, temperature, entropy, and cooling time. It produces several figures (X-ray spectra + best-fit models, MCMC corner plot, ICM profiles, cluster map) that can be used to check the validity of the analysis and to interpret the results.

Features
--------
* Uses the CIAO software and the CALDB database.
* Reprocesses the level 1 event files and remove flares from lightcurves.
* Identifies point sources with wavelet filters and mask them.
* Extracts the X-ray surface brightness profile in the 0.7-2.0 keV band.
* Estimates the ICM temperature profile or the mean ICM temperature through the analysis of the X-ray spectra.
* Estimates the ICM density profile from a Bayesian forward fit of the emission measure profile.
* Combines the ICM density and temperature profiles to estimate other ICM properties (entropy, pressure, cooling time, HSE mass)
* Produces adaptively-smoothed image of the cluster

Installation
------------
You first need to install the CIAO software and the CALDB database on your system:
https://cxc.cfa.harvard.edu/ciao/download/index.html
The best way to realize this installation is through conda.
Once your CIAO conda environment is installed you will need to add several packages.
Activate your new conda environment: conda activate ciao-"version number" <br />
Then run the following commands:

conda install --name ciao-"version number" termcolor astropy tqdm <br />
conda update --name ciao-"version number" matplotlib <br />
pip install gdpyc emcee pydl getdist 

The latest version of CIAO do not include the Chandra bakground files in the CALDB and you will need them for the analysis. <br />
Go to https://cxc.harvard.edu/ciao/download/caldb.html and download the latest version of the ACIS background event files. <br />
Unzip the file and move all fits files in CALDB/data/chandra/acis/bkgrnd/ <br />
You should find the CALDB folder in the ciao-"version number" folder created in the envs directory of anaconda3.

You will also need to set-up three environment variables in your .bash_profile, .bashrc, or .cshrc file: <br />
CXO_RES_DIR=/Directory where you want to save the results of your analyses <br />
CXO_PIPE=/Directory containing this pipeline (where the cxo_pipe_launch.py and param.py files are saved) <br />
CXO_BKG_FILES=/Directory containing the Chandra background files

Quickstart
----------
Once everything is installed you can go in your CXO_PIPE directory and run the following command:
```
ipython cxo_pipe_launch.py
```
The code will use the default param.py file and run the analysis of MACS J0329.6-0211, a cluster located at z=0.45. <br />
The whole analysis should take about 15 minutes. <br />
You can modify the param.py file to analyze other Chandra obsids with different definitions of the deprojection center.

Attribution
-----------
Please cite [F. Ruppin et al., Astrophys. J. 918, 43 (2021)] if you find this code useful in your research.
