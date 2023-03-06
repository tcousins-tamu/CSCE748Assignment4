# CSCE748Assignment4
# <ins> Directory Information </ins>
The *Code* directory contains all of the code used for this project including the unmodified **GetMask.py** and the modified **main.py** which contains the gradient domain image blending code. 

The *Images* directory contains all of the image masks, target images, and source images used for this implementation. This will not be present in the submission.

The *Results* directory contains all of the outputs from the program when it is run

# <ins> Running Instructions </ins>
Light modifications have been made over the original code. Instead of an "isMix" parameter, gradient mixes are made by adjusting the "alpha" parameter within the program. This change was made to adjust the Gradient Mix at will.
This change is reflected in the PoissonBlend and matrixParameters functions. The other large change is the addition of the matrixParameters function, which is used to create the parameters for constructing the sparse matrix used.
