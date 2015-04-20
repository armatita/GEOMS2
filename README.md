# GEOMS2
GEOMS2 is a scientific software for geosciences and geostatistics modeling. Provides interface for grid (mesh), point, surface and data (non-spatial) objects. It has a 3D viewer and 2D plots using the well known Python engines Mayavi and Matplotlib. It has several functions to manipulate your data as well as provide univariate and multivariate analysis.

![alt tag](/ART/DEFAULT/related/splash_screen.png?raw=true)

<h2>Reform branch</h2>
The reform branch is branch intended to build GEOMS2 almost from beginning. The arquitecture is different (check the architecture pdf file) and the dependencies also. It's our objective to change GUI, 2D and 3D engine in order to gain more performance and flexibility. We're still not sure what alternatives should be considered thought.

<h2>New architecture</h2>
I'm thinking of this folder hierarchy:


a) Launcher (file) ->  ui (folder)        -> gui stuff and embeded engines
                   ->  cerenalib (folder) -> the actual operational stuff

                   ->  art                -> icons, logos and buttons

<h2>Developer and contact</h2>

This software was developed within research center CERENA (IST - University of Lisbon).
![alt tag](/ART/cerena.png?raw=true)

You can contact us at: cerena.cmrp@gmail.com
