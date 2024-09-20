# machineLearningMagneticSensing
This is the code for the paper: "**Machine-learning based high-bandwidth magnetic sensing**" by Galya Haim, Stefano Martina, John Howell, Nir Bar-Gill and Filippo Caruso (2024). Preprint in [arXiv:2409.12820](https://arxiv.org/abs/2409.12820) ([https://doi.org/10.48550/arXiv.2409.12820](https://doi.org/10.48550/arXiv.2409.12820)).

* The folder `data` contains the experimental data.
* The folder `models` contains the Pytorch model for the neural network.
* The folder `sim` contains the Matlab files for the simulations.
  * `lorentzian_fit_lf.m` calculates the raster scans by fitting Lorentzian curves.
  * `mock_diamond2_new.m` and `mock_diamond2_new_galya.m` contains the simulation of the NV center (two different versions).
  * `monte_carlo_cs.m` and `monte_carlo_cs_galya.m` are used to generate simulated data (two different versions).
* `compareMLLorentzianFit.py` and `compareMLLorentzianFitHF.py` create data files for the plotting of the errors for the raster scans and ML models.
* `configurations.py` contains all the configurations for the various test runs.
* `createDataset.py`, `createDatasetMultiple.py` and `createDatasetSamplePlot.py` creates the simulated datasets.
* `funPlot.py` contains the functions for the plotting.
* `funPytorch.py` contains the function for the creation, training and evaluation of the ML models.
* `interpolateValidation.py` is used for the evaluation of raster scanning and ML with the interpolated data in the paper.
* `loaders.py` contains the pyTorch data loaders.
* `packDataset.py`, `packDatasetAugmented.py`, `packDatasetReal.py` and `packDatasetRealHF.py` pack various simulation files in one dataset.
* All the `plot*.py` are used to plot data and results.
* `runLinearRegression.py` is a baseline using linear regression.
* `runPredictions.py` is used to calculate the predictions of a trained model.
* `runPytorch.py` starts the training of a model (without hyperparameters optimization).
* `startTensorboard.py` is an utility to start Tensorboard of a specific configuration.
* `subsampleDataset.py` is used to subsample the data to less measurements.
* `tuneClean.py` is used to free space by cleaning the suboptimal runs within the hyperparameters optimization (Ray Tune) output directory.
* `tuneRunPytorch.py` starts the training of a model using the hyperparameters optimization utility (Ray Tune).
