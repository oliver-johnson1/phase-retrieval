# Comparision of Phase Retrieval Methods

This repository contains the code used to generate the results for the Stanford EE367 Project "Comparison of Phase Retrieval Methods".

### Organization
`methods/solver.py` contains the code used to solve the phase retrieval problem 

`examples` contains `samplePlot.py` and `switchPhases.py` which were used to generate the explanatory plots in the introduction of the report

`util/util.py` contains helper functions that make the code a bit cleaner

`test.py` is the main function where tests are run

`data` countains sample images from the BSDS300 and MNIST datasets

### Usage 
Activate your python environment and execute the following command to run the tests used to create the figures in the report and print out PSNRs values. 

``` sh
python test.py
```
By default this will run a small noise test on a subset of the tiger image, run the two methods for 10000 and 20000 iterations on the full tiger image, and run the methods on some samples from the MNIST dataset without and with oversampling.

test.py can be modified to change parameters (e.g. iterations, noise, input images etc) to run more tests

Images will be saved in `outputs`

### Acknowledgements
Part of the code was based on `hu-machine-learning/phase-retrieval-cgan` and the OSS implementation was based on the author's Matlab implementation [1].

[1] J. A. Rodriguez, R. Xu, C.-C. Chen, Y. Zou, and J. Miao, "Oversampling smoothness: an effective algorithm for phase retrieval of noisy diffraction intensities," Journal of Applied Crystallography, vol. 46, no. 2, p. 312–318, Feb 2013.
