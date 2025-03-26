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

[Dornheim et al. (2019)]: T. Dornheim, J. Vorberger, S. Groth, N. Hoffmann, Zh. A. Moldabekov, M. Bonitz; The static local field correction of the warm dense electron gas: An ab initio path integral Monte Carlo study and machine learning representation. J. Chem. Phys. 21 November 2019; 151 (19): 194104. https://doi.org/10.1063/1.5123013
