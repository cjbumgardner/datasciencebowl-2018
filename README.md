# datasciencebowl-2018
A collection of functions used for processing nuclei images and training a U-net to identify masks for nuclei from cell images. 

These files are intended mostly as examples of my work. I don't assume they would necessarily be useful for your personal 
use, but if you are interested, certainly inquire about further details on how to use the program. I'll will give a brief, 
mostly conceptual description below on the content of the programs. In the end, the design I used (specific hyperparameters)
was decent enough to place me in the 92nd percentile of the competetors. That is with no major post processing of the outcome
of the U-net. There's always next time! 


DSB2018_Unet_revised.py:
... is a u-net design written for tensorflow. The basic u-net design is 5 layers to "the bottom of the U". If you are unfamiliar with U-net designs, there are plenty of online references. The basic "vanilla" u-net design in this file has 3x3 convolution features, 2x2 max pooling, and 2x2 convolution-transpose features (or "up-convolution" features). There are parameters to change the number of convolution features at each level of the U, add dropout of features during training, add regularization, and several parameters of different types to manipulate the cost functions. 

The u-net takes as input a one channel image and outputs a 3 channel binary image. The 3 features where intended to be a layer of masks for interior of nuclei, boundary of nuclei, and for the background. The parameters for manipulating the cost function come in two types. One, a set of weight factors for the penalty given to incorrect categorization of interior, boundary, or exterior of nuclei (there is also a setting to give the boundary penalty to a fattened region around the boundary). The other factors are conceptually intented to give larger (or smaller) penalties for specific types of misclassification. For example, you might want it to be 10x worse for an exterior point to be mislabeled as a boundary than an interior point to be mislabeled as a boundary. These parameters were useful for identifying touching nuclei as separate, as well as getting the u-net to "care" more about the boundaries being labeled correctly.

datasciencebowl2018_revised.py:

This file is a collection all the other functions used to process the data. A general description of the process:
1) Identify training data the wasn't labeled well or has any other problems. Fix small problems in mask data.
2) Expand the "good" training data with small, random distortions of the original images.
3) Turn the maskes into masks for interior, exterior, and boundary of nuclei.
4) Normalize the 3 channel images into 1 channel images with various manipualtions for noise, contrast, weighting color channels
most promenent with "gradient activity", and etc. 
5) Cut the images into pieces of a prefered size for training the u-net. 

Then, post-process the data, convert the boundary/exterior/interior masks to "normal" masks, and convert masks to a 'run length encoded' csv file. 
                
