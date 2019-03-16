### Install relevant packages ###
For linux-64, a Anaconda spec file with all packages can be found at "spec-file.txt".

In order to create an environment from it, open a console in this directory and 
type "conda create --name myenv --file spec-file.txt". Then activate the new
environment by typing "cona activate myenv".

Alternatively, consult "spec-file.txt" for required packages and install them manually.
#################################


### LV Segmentation #############
1. To get started, copy the dataset into the folder "data/heart_scans".
2. Then, open a console in this directory and type "jupyter notebook". A browser window should open.
3. In the browser window, open "LV_Segmenter.ipynb" and follow instructions.

To illustrate the working of the program, a sample folder structure has been placed under "experiments/".
#################################