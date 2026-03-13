# Neural network interpolation of $S_{ii}$

Here is a (test) implementation of a neural network approach to interpolate
static structure factors over a range of temperatures, densities, ionizations
and scattering vectors. The work is motivated by the paper by 
[Dornheim et al. (2019)], where the authors interpolate very costly
simulations. We, here, use $S_{ii}$ calculated by HNC. Already here, the
speedup in inference is notable, allowing to front-load calculations, e.g., for
online analysis during an experiment.

We define the Neural network and a `jaxrts.Model` for the ionic scattering in
`model.py`. Data-generation, training and inference has been seperated into
individual files.

The `checkpoints/C` directory contains a net trained on carbon with 30,000
samples (split into training and validation 0.8/0.2), without any L2 norm and a
net architecture of (4 x 64 x 1024 x 1024 x 1). For the tested regime, the
agreement seems reasonable compared to a direct calculation of Sii.

## How to get started
### Generating training Data
1. Set up a jaxrts.PlasmaState with the elements and number fractions you like to create 
2. Decide whether you want to expand the ionization state of the PlasmaState to integer values
3. Set the range of conditions your interested and the total number of data points for which the HNC calcuations is
   performed
4. Generate the dataset - It is beeing saved in the train_data directory
5. The name of the .h5 file contains the information about the number fractions the total number of data points and if
   the state is expanded or not to create the HNC dataset

### Training a Neural Net
6. To train a neural net you just have to set the file_path to the just created HNC training data
7. The code will use the file_path to set important parameters and to select the correct dataset class corresponding to
   the elements. Currently for the non expanded case 1,2,3 and 4 Component PlasmaStates can be trained and for the
   expanded state 1 or 2 component PlasmaState. The user can easily add new Classes if needed for their liking
8. The number of hidden layers can be adjusted, but a structure of [64, 128, 1024] proved to be decent at the task of
   interpolating the Sii's.
9. Once everything is properly set. The total number of training epoches can be set - 2000 is a good start for testing.
10. Once the training is running every 100 epoches a safestate is safed in the `checkpoints` dir. The directory are
    labeled as "100_epoch", "200_epoch", ... and so on
11. One just have to copy/move the NN directory for a given epoche (typically one uses the last epoche) and give it a proper name
12. This is your trained NN. A loss.png figure is plotted to sowcase if the NN is converged.

### Infer/ Test the NN against HNC calculations
13. Once the NN is trained we want to test how it performs against new calculated values that are not present in the
    training data - so to say against newly calcuated HNC output for a given condition. In the infer script one has to
    set the file_path to the PlasmaState which was saved in generate data as a .json file in the train_data dir.
14. Set the directory path for the safed NN which you just trained
15. Hit enter and 3D plots are created that compare the output of the NN (plotted as orange grid) against the output of
    the HNC calculation (blue crosses)

[Dornheim et al. (2019)]: T. Dornheim, J. Vorberger, S. Groth, N. Hoffmann, Zh. A. Moldabekov, M. Bonitz; The static local field correction of the warm dense electron gas: An ab initio path integral Monte Carlo study and machine learning representation. J. Chem. Phys. 21 November 2019; 151 (19): 194104. https://doi.org/10.1063/1.5123013
