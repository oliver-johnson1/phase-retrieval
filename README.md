# Comparision of Phase Retrieval Methods

This repository contains the code used to generate the results for the Stanford EE367 Project "Comparison of Phase Retrieval Methods".

## Organization
methods/solver.py contains the code used to solve the phase retrieval problem 
examples contains samplePlot.py and switchPhases.py which were used to generate the explanatory plots in the introduction of the report
util/util.py contains helper functions that make the code a bit cleaner
test.py is the main function where tests are run
data countains sample images from the BSDS300 and MNIST datasets

## Usage 
Activate your python environment and execute the following command to run the tests used to create the figure in the report. 

``` sh
python test.py
```

test.py can be modified to change parameters (e.g. iterations, noise, input images etc) to run more tests
