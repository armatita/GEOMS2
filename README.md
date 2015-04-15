# GEOMS2
GEOMS2 is a scientific software for geosciences and geostatistics modeling. Provides interface for grid (mesh), point, surface and data (non-spatial) objects. It has a 3D viewer and 2D plots using the well known Python engines Mayavi and Matplotlib. It has several functions to manipulate your data as well as provide univariate and multivariate analysis.

![alt tag](/ART/DEFAULT/related/splash_screen.png?raw=true)

<h2>Information for users</h2>
Check documentation folder to see a pdf file with some guidelines on how to use this software.

<h2>Information for developers</h2>
GEOMS2.py is the main python script which than depends on cerena_file_utils, cerena_grid_utils, cerena_multivariate_utils, cerena_object, cerena_plugins, hardworklib, pymayalibrary and pympllibrary. This is pretty messy and needs some deep code refactoring at this time.

<h2>Launch this code</h2>
If you want to run GEOMS2 you'll need WinPython or comparable distribution. Currently the official version of GEOMS2 comes with a batch file (this software launcher) that guarantees that GEOMS2.py is being run by the WinPython portable version. Our batch looks like this:

@echo off
start WinPython-64bit-2.7.9.3\python-2.7.9.amd64\python.exe GEOMS2.py -wo %CD%

<h2>Developer and contact</h2>

This software was developed within research center CERENA (IST - University of Lisbon).
![alt tag](/ART/cerena.png?raw=true)

You can contact us at: cerena.cmrp@gmail.com
