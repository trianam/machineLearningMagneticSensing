#     tuneConfigurations.py
#     The configurations for all the experiments.
#     Copyright (C) 2021  Stefano Martina (stefano.martina@unifi.it)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import math
from conf import Conf
from ray import tune
import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope
from collections import defaultdict


configGlobal = Conf({
    "datasetsDir":          'datasets',
    "dataset":              'dataset1.npz',
    "splits":               ['train', 'valid', 'test'],
    "debug":                False,
    "force":                False,
    "taskType":             "regressionL2", #classification, regressionL1, regressionL2, regressionMRE
    "nonVerbose":           False,
    "useTune":              False,
    "tuneHyperOpt":         True,
    "tuneNumSamples":       1000, #if tuneHyperOpt then it is total number, else it is number for each point in grid
    "tuneParallelProc":     10,
    "tuneGrace":            2,
    "tuneGpuNum":           0,
    "optimizer":            'adam', #sgdtuneGpuNum or adam
    "learningRate":         0.001,
    "dimX":                 100,
    "dimY":                 1,
    "normalizeX":           False,
    "normalizeY":           True,
    "minMax":               ((90, 120),), #from data creation
    "model":                'modelMLP',
    "activation":           'relu',
    "finalActivation":      'sigmoid',
    "hiddenLayers":         2,
    "hiddenDim":            950,
    "dropout":              0,
    "weightDecay":          0,
    "batchSize":            16,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                          #None or patience
    "tensorBoard":          True,
    "logCurves":            True,
    "logEveryBatch":        False,
    "modelSave":            "best",
    "bestKey":              "mseL",
    "plotMetric":           "mae",
    "plotLossName":         "MSE (normalized)",
    "plotMetricName":       "MAE",
    "bestSign":             "<",
    "modelLoad":            'last.pt',
    "shuffleDataset":       True,
    "filePredAppendix":     None,
})


# ================================================
run1 = configGlobal.copy({
    "path":                 'run1',
})

run2 = configGlobal.copy({
    "path":                 'run2',
    "dataset":              'dataset4.npz',
    "dimX":                 50,
})

run3 = configGlobal.copy({
    "path":                 'run3',
    "dataset":              'dataset2.npz',
})

run4 = configGlobal.copy({
    "path":                 'run4',
    "dataset":              'dataset3.npz',
})


run5 = configGlobal.copy({
    "path":                 'run5',
    "dataset":              'dataset5.npz',
    "dimX":                 34,
})

# ================================================

trun1 = configGlobal.copy({
    "path":                 'trun1',
    "dataset":              'dataset2.npz',

    "useTune":              True,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        # "dropout":              tune.choice([0, 0.2, 0.5]),
        # "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
})

trun2 = configGlobal.copy({
    "path": 'trun2',
    "dataset": 'dataset4.npz',
    "dimX": 50,

    "useTune": True,
    "tuneConf": {
        "learningRate": tune.choice([1e-2, 1e-3, 1e-4]),
        # "dropout":              tune.choice([0, 0.2, 0.5]),
        # "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize": tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers": tune.lograndint(1, 32),
        "hiddenDim": tune.lograndint(1, 1024),
    },
})

# ================================================

frun1 = configGlobal.copy({
    "path":                 'frun1',
    "dataset":              'dataset2f.npz',
    "dimY":                 3,
    "minMax":               ((90, 120), (30, 34), (17, 21)),  # from data creation
    "learningRate":         0.0001,
    "batchSize":            4,
    "hiddenLayers":         4,
    "hiddenDim":            1013,
})

# ================================================

fgrun1 = configGlobal.copy({
    "path":                 'fgrun1',
    "dataset":              'datasetFG1.npz',
    "dimY":                 3,
    "dimX":                 421,
    "minMax":               ((80, 120), (10, 35), (0, 30)),  # from data creation
    "learningRate":         0.0001,
    "batchSize":            4,
    "hiddenLayers":         4,
    "hiddenDim":            1013,
})

# ================================================

fgsfrun1 = configGlobal.copy({
    "path":                 'fgsfrun1',
    "dataset":              'datasetFGSF1.npz',
    "dimY":                 3,
    "dimX":                 246,
    "minMax":               ((20, 60), (0, 90), (0, 90)),  # from data creation
    "learningRate":         0.0001,
    "batchSize":            4,
    "hiddenLayers":         4,
    "hiddenDim":            1013,
})
fgsfrun1c = fgsfrun1.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               600,
    "modelLoad":            'last.pt',
})
fgsfrun1p = fgsfrun1.copy({ # to predict on real data
    "dataset":              'datasetRGSF1.npz',
    "splits":               ['real'],
    # "modelLoad":            'best.pt',
    "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------
fgsfrun2 = fgsfrun1.copy({
    "path":                 'fgsfrun2',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun2p = fgsfrun2.copy({ # to predict on real data
    "dataset":              'datasetRGSF1.npz',
    "splits":               ['real'],
    # "modelLoad":            'best.pt',
    "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#--------------------------- real data validation
fgsfrun3 = fgsfrun1.copy({
    "path":                 'fgsfrun3',
    "dataset":              'datasetFRGSF1.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun3c = fgsfrun3.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               500,
    "modelLoad":            'last.pt',
})
fgsfrun3p = fgsfrun3.copy({ # to predict on real data
    "dataset":              'datasetRGSF1.npz',
    "splits":               ['real'],
    # "modelLoad":            'best.pt',
    "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

fgsfrun4 = fgsfrun3.copy({
    "path":                 'fgsfrun4',
    "hiddenLayers":         2,
})

#---------------------------- small range
fgsfrun5 = fgsfrun1.copy({
    "path":                 'fgsfrun5',
    "dataset":              'datasetFRGSF1v2.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun5c = fgsfrun5.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun5p = fgsfrun5.copy({ # to predict on real data
    "dataset":              'datasetRGSF1.npz',
    "splits":               ['real'],
    # "modelLoad":            'best.pt',
    "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------- small range subsampled 2
fgsfrun6 = fgsfrun1.copy({
    "path":                 'fgsfrun6',
    "dataset":              'datasetFRGSF1v2s2.npz',
    "dimX":                 123,
    "epochs":               1000,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun6c = fgsfrun6.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               100,
    "modelLoad":            'last.pt',
})
fgsfrun6p = fgsfrun6.copy({ # to predict on real data
    "dataset":              'datasetRGSF1s2.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------- small range subsampled 3
fgsfrun7 = fgsfrun1.copy({
    "path":                 'fgsfrun7',
    "dataset":              'datasetFRGSF1v2s3.npz',
    "dimX":                 82,
    "epochs":               1000,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun7c = fgsfrun7.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               100,
    "modelLoad":            'last.pt',
})
fgsfrun7p = fgsfrun7.copy({ # to predict on real data
    "dataset":              'datasetRGSF1s3.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------- small range subsampled 4
fgsfrun8 = fgsfrun1.copy({
    "path":                 'fgsfrun8',
    "dataset":              'datasetFRGSF1v2s4.npz',
    "dimX":                 62,
    "epochs":               1000,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun8c = fgsfrun8.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               100,
    "modelLoad":            'last.pt',
})
fgsfrun8p = fgsfrun8.copy({ # to predict on real data
    "dataset":              'datasetRGSF1s4.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------- small range subsampled 5
fgsfrun9 = fgsfrun1.copy({
    "path":                 'fgsfrun9',
    "dataset":              'datasetFRGSF1v2s5.npz',
    "dimX":                 50,
    "epochs":               1000,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun9c = fgsfrun9.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               100,
    "modelLoad":            'last.pt',
})
fgsfrun9p = fgsfrun9.copy({ # to predict on real data
    "dataset":              'datasetRGSF1s5.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------- small range subsampled 6
fgsfrun10 = fgsfrun1.copy({
    "path":                 'fgsfrun10',
    "dataset":              'datasetFRGSF1v2s6.npz',
    "dimX":                 41,
    "epochs":               1000,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun10c = fgsfrun10.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               100,
    "modelLoad":            'last.pt',
})
fgsfrun10p = fgsfrun10.copy({ # to predict on real data
    "dataset":              'datasetRGSF1s6.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------- small range subsampled 7
fgsfrun11 = fgsfrun1.copy({
    "path":                 'fgsfrun11',
    "dataset":              'datasetFRGSF1v2s7.npz',
    "dimX":                 36,
    "epochs":               1000,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun11c = fgsfrun11.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               100,
    "modelLoad":            'last.pt',
})
fgsfrun11p = fgsfrun11.copy({ # to predict on real data
    "dataset":              'datasetRGSF1s7.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------------------- small range subsampled 8
fgsfrun12 = fgsfrun1.copy({
    "path":                 'fgsfrun12',
    "dataset":              'datasetFRGSF1v2s8.npz',
    "dimX":                 31,
    "epochs":               1000,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
})
fgsfrun12c = fgsfrun12.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               100,
    "modelLoad":            'last.pt',
})
fgsfrun12p = fgsfrun12.copy({ # to predict on real data
    "dataset":              'datasetRGSF1s8.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#================================== peak predictions
fgsfrun13 = fgsfrun1.copy({
    "path":                 'fgsfrun13',
    "dataset":              'datasetFRGSF1v2P.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13c = fgsfrun13.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13p = fgsfrun13.copy({ # to predict on real data
    "dataset":              'datasetRGSF1P.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------- subsampled 2
fgsfrun13s2 = fgsfrun1.copy({
    "path":                 'fgsfrun13s2',
    "dataset":              'datasetFRGSF1v2Ps2.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 123,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13s2c = fgsfrun13s2.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13s2p = fgsfrun13s2.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps2.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})
#---------------- subsampled 3
fgsfrun13s3 = fgsfrun1.copy({
    "path":                 'fgsfrun13s3',
    "dataset":              'datasetFRGSF1v2Ps3.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 82,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13s3c = fgsfrun13s3.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13s3p = fgsfrun13s3.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps3.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})
#---------------- subsampled 4
fgsfrun13s4 = fgsfrun1.copy({
    "path":                 'fgsfrun13s4',
    "dataset":              'datasetFRGSF1v2Ps4.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 62,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13s4c = fgsfrun13s4.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13s4p = fgsfrun13s4.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps4.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})
#---------------- subsampled 5
fgsfrun13s5 = fgsfrun1.copy({
    "path":                 'fgsfrun13s5',
    "dataset":              'datasetFRGSF1v2Ps5.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 50,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13s5c = fgsfrun13s5.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13s5p = fgsfrun13s5.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps5.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})
#---------------- subsampled 6
fgsfrun13s6 = fgsfrun1.copy({
    "path":                 'fgsfrun13s6',
    "dataset":              'datasetFRGSF1v2Ps6.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 41,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13s6c = fgsfrun13s6.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13s6p = fgsfrun13s6.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps6.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})
#---------------- subsampled 7
fgsfrun13s7 = fgsfrun1.copy({
    "path":                 'fgsfrun13s7',
    "dataset":              'datasetFRGSF1v2Ps7.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 36,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13s7c = fgsfrun13s7.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13s7p = fgsfrun13s7.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps7.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})
#---------------- subsampled 8
fgsfrun13s8 = fgsfrun1.copy({
    "path":                 'fgsfrun13s8',
    "dataset":              'datasetFRGSF1v2Ps8.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 31,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun13s8c = fgsfrun13s8.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun13s8p = fgsfrun13s8.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps8.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})


fgsfrun14 = fgsfrun1.copy({ #big ranges
    "path":                 'fgsfrun14',
    "dataset":              'datasetFRGSF1P.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun14c = fgsfrun14.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun14p = fgsfrun14.copy({ # to predict on real data
    "dataset":              'datasetRGSF1P.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

fgsfrun15 = fgsfrun1.copy({ #MAE loss
    "path":                 'fgsfrun15',
    "dataset":              'datasetFRGSF1v2P.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimY":                 8,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "taskType":             "regressionL1",  # classification, regressionL1, regressionL2, regressionMRE
    "bestKey":              "maeL",
    "plotLossName":         "MAE",
})
fgsfrun15c = fgsfrun15.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun15p = fgsfrun15.copy({ # to predict on real data
    "dataset":              'datasetRGSF1P.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#================================== peak predictions, realistic noise and X normalization
fgsfrun16 = fgsfrun1.copy({
    "path":                 'fgsfrun16',
    "dataset":              'datasetFRGSF1v2nP.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
    # "modelLoad":            'best.pt',
    "modelLoad":            'last.pt',
})
fgsfrun16c = fgsfrun16.copy({ # to continue training
    "startEpoch":           -1,
    "epochs":               800,
    "modelLoad":            'last.pt',
})
fgsfrun16p = fgsfrun16.copy({ # to predict on real data
    "dataset":              'datasetRGSF1P.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

# with TUNE
# best: {'learningRate': 0.01, 'dropout': 0, 'weightDecay': 1e-06, 'batchSize': 4, 'hiddenLayers': 16, 'hiddenDim': 159}
tfgsfrun16 = fgsfrun16.copy({
    "path":                 'tfgsfrun16',
    "useTune":              True,
    # "modelLoad":            'best.pt',
    "modelLoad":            'last.pt',
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
})
tfgsfrun16p = tfgsfrun16.copy({ # to predict on real data
    "dataset":              'datasetRGSF1P.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------- subsampled 8

fgsfrun16s8 = fgsfrun1.copy({
    "path":                 'fgsfrun16s8',
    "dataset":              'datasetFRGSF1v2nPs8.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 31,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun16s8p = fgsfrun16s8.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps8.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------- subsampled 9

fgsfrun16s9 = fgsfrun1.copy({
    "path":                 'fgsfrun16s9',
    "dataset":              'datasetFRGSF1v2nPs9.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 28,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun16s9p = fgsfrun16s9.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps9.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------- subsampled 10

fgsfrun16s10 = fgsfrun1.copy({
    "path":                 'fgsfrun16s10',
    "dataset":              'datasetFRGSF1v2nPs10.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 25,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun16s10p = fgsfrun16s10.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps10.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#---------------- subsampled 11

fgsfrun16s11 = fgsfrun1.copy({
    "path":                 'fgsfrun16s11',
    "dataset":              'datasetFRGSF1v2nPs11.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimX":                 23,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "plotLossName":         "MSE",
})
fgsfrun16s11p = fgsfrun16s11.copy({ # to predict on real data
    "dataset":              'datasetRGSF1Ps11.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

# mae loss
fgsfrun16l1 = fgsfrun1.copy({
    "path":                 'fgsfrun16l1',
    "dataset":              'datasetFRGSF1v2nP.npz',
    "epochs":               200,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           False,
    "minMax":               None,
    "finalActivation":      'none',
    "taskType":             "regressionL1",
    "bestKey":              "maeL",
    "plotLossName":         "MAE",
})
fgsfrun16l1p = fgsfrun16l1.copy({ # to predict on real data
    "dataset":              'datasetRGSF1P.npz',
    "splits":               ['real'],
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
    "batchSize":            1,
    "shuffleDataset":       False,
})

#================================== peak predictions, realistic noise and X normalization synth validation y normalization
fgsfrun17 = fgsfrun1.copy({
    "path":                 'fgsfrun17',
    "dataset":              'datasetFGSF1v2nP.npz',
    "epochs":               100,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            512,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2744.9405479334164, 2819.1908696379664),
                            (2774.953648417945, 2832.1134864432233),
                            (2812.326530507976, 2851.021438717914),
                            (2840.330350271225, 2864.993463765968),
                            (2889.006536234032, 2913.669649728775),
                            (2902.978561282086, 2941.673469492024),
                            (2921.8865135567767, 2979.046351582055),
                            (2934.8091303620336, 3009.0594520665836)], #out of boundaries [2754,3000]
    "finalActivation":      'none',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

fgsfrun17s2 = fgsfrun17.copy({
    "path":                 'fgsfrun17s2',
    "dataset":              'datasetFGSF1v2nPs2.npz',
    "dimX":                 123,
})
fgsfrun17s3 = fgsfrun17.copy({
    "path":                 'fgsfrun17s3',
    "dataset":              'datasetFGSF1v2nPs3.npz',
    "dimX":                 82,
})
fgsfrun17s4 = fgsfrun17.copy({
    "path":                 'fgsfrun17s4',
    "dataset":              'datasetFGSF1v2nPs4.npz',
    "dimX":                 62,
})
fgsfrun17s5 = fgsfrun17.copy({
    "path":                 'fgsfrun17s5',
    "dataset":              'datasetFGSF1v2nPs5.npz',
    "dimX":                 50,
})
fgsfrun17s6 = fgsfrun17.copy({
    "path":                 'fgsfrun17s6',
    "dataset":              'datasetFGSF1v2nPs6.npz',
    "dimX":                 41,
})
fgsfrun17s7 = fgsfrun17.copy({
    "path":                 'fgsfrun17s7',
    "dataset":              'datasetFGSF1v2nPs7.npz',
    "dimX":                 36,
})
fgsfrun17s8 = fgsfrun17.copy({
    "path":                 'fgsfrun17s8',
    "dataset":              'datasetFGSF1v2nPs8.npz',
    "dimX":                 31,
})
fgsfrun17s9 = fgsfrun17.copy({
    "path":                 'fgsfrun17s9',
    "dataset":              'datasetFGSF1v2nPs9.npz',
    "dimX":                 28,
})
fgsfrun17s10 = fgsfrun17.copy({
    "path":                 'fgsfrun17s10',
    "dataset":              'datasetFGSF1v2nPs10.npz',
    "dimX":                 25,
})
fgsfrun17s11 = fgsfrun17.copy({
    "path":                 'fgsfrun17s11',
    "dataset":              'datasetFGSF1v2nPs11.npz',
    "dimX":                 23,
})
fgsfrun17s12 = fgsfrun17.copy({
    "path":                 'fgsfrun17s12',
    "dataset":              'datasetFGSF1v2nPs12.npz',
    "dimX":                 21,
})
fgsfrun17s13 = fgsfrun17.copy({
    "path":                 'fgsfrun17s13',
    "dataset":              'datasetFGSF1v2nPs13.npz',
    "dimX":                 19,
})

#================================== peak predictions, High Field data, y normalization
fgsfrun18 = configGlobal.copy({
    "path":                 'fgsfrun18',
    "dataset":              'datasetFRGHF1P.npz',
    "epochs":               100,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            408,
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

for s in range(2,41):
    inDim = math.ceil(600/s)
    tmpConf = fgsfrun18.copy({
        "path": f'fgsfrun18s{s}',
        "dataset": f'datasetFRGHF1Ps{s}.npz',
        "dimX": inDim,
        "hiddenDim": min(inDim, math.ceil(inDim/3*2 + 8))
    })
    exec(f"fgsfrun18s{s} = tmpConf")

for s in ['']+[f's{s}' for s in range(2,14)]:
    exec(f"fgsfrun18{s}L = fgsfrun18{s}.copy({{'modelLoad':'last.pt'}})")

#================================== peak predictions, High Field data, y normalization, augmented dataset
fgsfrun19 = configGlobal.copy({
    "path":                 'fgsfrun19',
    "dataset":              'datasetAHF1P.npz',
    "epochs":               100,
    "learningRate":         0.001,
    "batchSize":            4,
    "hiddenLayers":         1,
    "hiddenDim":            408,
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    # 'extraWait':            1,
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

for s in range(2,41):
    inDim = math.ceil(600/s)
    tmpConf = fgsfrun19.copy({
        "path": f'fgsfrun19s{s}',
        "dataset": f'datasetAHF1Ps{s}.npz',
        "dimX": inDim,
        "hiddenDim": min(inDim, math.ceil(inDim/3*2 + 8))
    })
    exec(f"fgsfrun19s{s} = tmpConf")

for s in ['']+[f's{s}' for s in range(2,14)]:
    exec(f"fgsfrun19{s}L = fgsfrun19{s}.copy({{'modelLoad':'last.pt'}})")

#================================== peak predictions, High Field data, y normalization, only real dataset, tune opt
# ATTENTION: was run with only real data
# best after run: batchSize=4,dropout=0,hiddenDim=9,hiddenLayers=3,learningRate=0.0100,weightDecay=0
# best analysis : batchSize=4,dropout=0,hiddenDim=9,hiddenLayers=3,learningRate=0.0100,weightDecay=0
fgsfrun19t = configGlobal.copy({
    "path":                 'fgsfrun19t',
    "dataset":              'datasetAHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       10000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#best config
fgsfrun19b = configGlobal.copy({
    "path":                 'fgsfrun19b',
    "dataset":              'datasetAHF1P.npz',
    "epochs":               1000,
    "learningRate":         1e-2,
    "dropout":              0,
    "weightDecay":          0,
    "batchSize":            4,
    "hiddenLayers":         3,
    "hiddenDim":            9,
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#alternate real shuffling (works better than 19b)
fgsfrun19b3 = configGlobal.copy({
    "path":                 'fgsfrun19b3',
    "dataset":              'datasetA3HF1P.npz',
    "epochs":               1000,
    "learningRate":         1e-2,
    "dropout":              0,
    "weightDecay":          0,
    "batchSize":            4,
    "hiddenLayers":         3,
    "hiddenDim":            9,
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#================================== peak predictions, High Field data, y normalization, augmented dataset, tune opt
#works worst than 19b
fgsfrun20 = configGlobal.copy({
    "path":                 'fgsfrun20',
    "dataset":              'datasetA2HF1P.npz',
    "epochs":               100,
    "learningRate":         1e-2,
    "dropout":              0,
    "weightDecay":          0,
    "batchSize":            4,
    "hiddenLayers":         3,
    "hiddenDim":            9,
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#also worst than 19b
fgsfrun20b = configGlobal.copy({
    "path":                 'fgsfrun20b',
    "dataset":              'datasetA4HF1P.npz',
    "epochs":               100,
    "learningRate":         1e-2,
    "dropout":              0,
    "weightDecay":          0,
    "batchSize":            4,
    "hiddenLayers":         3,
    "hiddenDim":            9,
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})


#================================== peak predictions, High Field data, y normalization, augmented dataset, tune opt
# best after run: batchSize=2,dropout=0,hiddenDim=1020,hiddenLayers=7,learningRate=0.0001,weightDecay=0
# best analysis : batchSize=2,dropout=0,hiddenDim=551,hiddenLayers=6,learningRate=0.0001,weightDecay=0
fgsfrun20t = configGlobal.copy({
    "path":                 'fgsfrun20t',
    "dataset":              'datasetA4HF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

for s in range(2,41):
    inDim = math.ceil(600/s)
    tmpConf = configGlobal.copy({
        "path": f'fgsfrun20ts{s}',
        "dimX": inDim,
        "dataset": f'datasetA4HF1Ps{s}.npz',
        "epochs": 100,
        "learningRate": 0.0001,
        "dropout": 0,
        "weightDecay": 0,
        "batchSize": 2,
        "hiddenLayers": 6,
        "hiddenDim": 551,
        "dimY": 8,
        "normalizeX": True,
        "normalizeY": True,
        "minMax": [(2530, 3250)] * 8,
        "finalActivation": 'sigmoid',
        "plotLossName": "MSE",
        "modelLoad": 'best.pt',
        # "modelLoad":            'last.pt',
    })
    exec(f"fgsfrun20ts{s} = tmpConf")


# best after run: batchSize=4,dropout=0,hiddenDim=281,hiddenLayers=3,learningRate=0.0100,weightDecay=0
# best analysis : batchSize=4,dropout=0,hiddenDim=132,hiddenLayers=4,learningRate=0.0100,weightDecay=0
fgsfrun21t = configGlobal.copy({
    "path":                 'fgsfrun21t',
    "dataset":              'datasetA3HF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

# best after run: batchSize=4,dropout=0,hiddenDim=61,hiddenLayers=3,learningRate=0.0010,weightDecay=0
# best analysis : batchSize=4,dropout=0,hiddenDim=39,hiddenLayers=4,learningRate=0.0010,weightDecay=0
fgsfrun22t = configGlobal.copy({
    "path":                 'fgsfrun22t',
    "dataset":              'datasetA2HF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       10000, #mistake
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#================================== peak predictions, High Field data, y normalization, augmented dataset, interpolation, tune opt
fgsfrun23ts2 = configGlobal.copy({
    "path":                 'fgsfrun23ts2',
    "dataset":              'datasetIA4HF1Ps2.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

for s in list(range(3,26))+[27,28,29,30,32,34,36,38,40]: #to avoid doubles with same number of measurements
    tmpConf = fgsfrun23ts2.copy({
        "path": f'fgsfrun23ts{s}',
        "dataset": f'datasetIA4HF1Ps{s}.npz',
    })
    exec(f"fgsfrun23ts{s} = tmpConf")

# for s in list(range(13,26))+[27,28,29,30,32,34,36,38,40]: # use best hyperparams of fgsfrun23ts12
for s in list(range(10,26))+[27,28,29,30,32,34,36,38,40]: # use best hyperparams of fgsfrun23ts12
    tmpConf = fgsfrun23ts2.copy({
        "path":         f'fgsfrun23s{s}',
        "dataset":      f'datasetIA4HF1Ps{s}.npz',
        "useTune":      False,
        "nonVerbose":   False,
        "learningRate": 0.0001,
        "dropout":      0,
        "weightDecay":  0,
        "batchSize":    2,
        "hiddenLayers": 8,
        "hiddenDim":    1019,
    })
    exec(f"fgsfrun23s{s} = tmpConf")


#================================== peak predictions, High Field data, y normalization, real dataset, tune opt, shuffled
# best after run:
# best analysis : batchSize=4,dropout=0,hiddenDim=58,hiddenLayers=4,learningRate=0.0100,weightDecay=0
fgsfrun24t = configGlobal.copy({
    "path":                 'fgsfrun24t',
    "dataset":              'datasetA5HF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

# for s in range(2,41):
#     inDim = math.ceil(600/s)
#     tmpConf = configGlobal.copy({
#         "path": f'fgsfrun24ts{s}',
#         "dimX": inDim,
#         "dataset": f'datasetA5HF1Ps{s}.npz',
#         "epochs": 100,
#         "learningRate": 0.0001,
#         "dropout": 0,
#         "weightDecay": 0,
#         "batchSize": 2,
#         "hiddenLayers": 6,
#         "hiddenDim": 551,
#         "dimY": 8,
#         "normalizeX": True,
#         "normalizeY": True,
#         "minMax": [(2530, 3250)] * 8,
#         "finalActivation": 'sigmoid',
#         "plotLossName": "MSE",
#         "modelLoad": 'best.pt',
#         # "modelLoad":            'last.pt',
#     })
#     exec(f"fgsfrun24ts{s} = tmpConf")

#================================== peak predictions, High Field data, y normalization, augmented dataset, tune opt, shuffled
# best after run: batchSize=8,dropout=0,hiddenDim=421,hiddenLayers=4,learningRate=0.0010,weightDecay=0
# best analysis : batchSize=8,dropout=0,hiddenDim=381,hiddenLayers=4,learningRate=0.0010,weightDecay=0
fgsfrun25t = configGlobal.copy({
    "path":                 'fgsfrun25t',
    "dataset":              'datasetA6HF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

# for s in range(2,41):
#     inDim = math.ceil(600/s)
#     tmpConf = configGlobal.copy({
#         "path": f'fgsfrun25ts{s}',
#         "dimX": inDim,
#         "dataset": f'datasetA6HF1Ps{s}.npz',
#         "epochs": 100,
#         "learningRate": 0.0001,
#         "dropout": 0,
#         "weightDecay": 0,
#         "batchSize": 2,
#         "hiddenLayers": 6,
#         "hiddenDim": 551,
#         "dimY": 8,
#         "normalizeX": True,
#         "normalizeY": True,
#         "minMax": [(2530, 3250)] * 8,
#         "finalActivation": 'sigmoid',
#         "plotLossName": "MSE",
#         "modelLoad": 'best.pt',
#         # "modelLoad":            'last.pt',
#     })
#     exec(f"fgsfrun25ts{s} = tmpConf")

#================================== peak predictions, High Field data, y normalization, augmented dataset 10k, tune opt, split 1
# best after run:
# best analysis :
fgsfrun26t = configGlobal.copy({
    "path":                 'fgsfrun26t',
    "dataset":              'datasetA2bHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

# for s in range(2,41):
#     inDim = math.ceil(600/s)
#     tmpConf = configGlobal.copy({
#         "path": f'fgsfrun26ts{s}',
#         "dimX": inDim,
#         "dataset": f'datasetA2bHF1Ps{s}.npz',
#         "epochs": 100,
#         "learningRate": 0.0001,
#         "dropout": 0,
#         "weightDecay": 0,
#         "batchSize": 2,
#         "hiddenLayers": 6,
#         "hiddenDim": 551,
#         "dimY": 8,
#         "normalizeX": True,
#         "normalizeY": True,
#         "minMax": [(2530, 3250)] * 8,
#         "finalActivation": 'sigmoid',
#         "plotLossName": "MSE",
#         "modelLoad": 'best.pt',
#         # "modelLoad":            'last.pt',
#     })
#     exec(f"fgsfrun26ts{s} = tmpConf")

#================================== peak predictions, High Field data, y normalization, augmented dataset 10k, tune opt, split 2
# best after run:
# best analysis :
fgsfrun27t = configGlobal.copy({
    "path":                 'fgsfrun27t',
    "dataset":              'datasetA4bHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

# for s in range(2,41):
#     inDim = math.ceil(600/s)
#     tmpConf = configGlobal.copy({
#         "path": f'fgsfrun27ts{s}',
#         "dimX": inDim,
#         "dataset": f'datasetA4bHF1Ps{s}.npz',
#         "epochs": 100,
#         "learningRate": 0.0001,
#         "dropout": 0,
#         "weightDecay": 0,
#         "batchSize": 2,
#         "hiddenLayers": 6,
#         "hiddenDim": 551,
#         "dimY": 8,
#         "normalizeX": True,
#         "normalizeY": True,
#         "minMax": [(2530, 3250)] * 8,
#         "finalActivation": 'sigmoid',
#         "plotLossName": "MSE",
#         "modelLoad": 'best.pt',
#         # "modelLoad":            'last.pt',
#     })
#     exec(f"fgsfrun27ts{s} = tmpConf")

#================================== peak predictions, High Field data, y normalization, synth dataset 93, tune opt, split 2 to emulate only real
# best after run: batchSize=2,dropout=0.2000,hiddenDim=529,hiddenLayers=6,learningRate=0.0100,weightDecay=0.0010
# best analysis : batchSize=2,dropout=0,hiddenDim=747,hiddenLayers=5,learningRate=0.0100,weightDecay=0.0010
fgsfrun28t = configGlobal.copy({
    "path":                 'fgsfrun28t',
    "dataset":              'datasetA7HF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#================================== peak predictions, High Field data, y normalization, synth dataset 93, tune opt, split 2 to emulate only real, same valid test
# best after run: batchSize=8,dropout=0.2000,hiddenDim=393,hiddenLayers=3,learningRate=0.0100,weightDecay=0
# best analysis : batchSize=8,dropout=0.2000,hiddenDim=222,hiddenLayers=3,learningRate=0.0100,weightDecay=0
fgsfrun29t = configGlobal.copy({
    "path":                 'fgsfrun29t',
    "dataset":              'datasetA7bHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#================================== peak predictions, High Field data, y normalization, only real, tune opt, split 2, different valid test
# best after run: batchSize=32,dropout=0.2000,hiddenDim=679,hiddenLayers=4,learningRate=0.0010,weightDecay=0
# best analysis : batchSize=32,dropout=0.2000,hiddenDim=909,hiddenLayers=4,learningRate=0.0010,weightDecay=0
fgsfrun30t = configGlobal.copy({
    "path":                 'fgsfrun30t',
    "dataset":              'datasetA3tHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#================================== peak predictions, High Field data, y normalization, augmented dataset, tune opt, split 2, different valid test
# best after run: batchSize=8,dropout=0,hiddenDim=427,hiddenLayers=4,learningRate=0.0010,weightDecay=0
# best analysis : batchSize=8,dropout=0,hiddenDim=228,hiddenLayers=5,learningRate=0.0010,weightDecay=0
fgsfrun31t = configGlobal.copy({
    "path":                 'fgsfrun31t',
    "dataset":              'datasetA4tHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     15,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})



#================================== peak predictions, High Field data, y normalization, synth 10k dataset, tune opt
# best after run: batchSize=8,dropout=0,hiddenDim=861,hiddenLayers=7,learningRate=0.0001,weightDecay=0
# best analysis : 
fgsfrun32t = configGlobal.copy({
    "path":                 'fgsfrun32t',
    # "dataset":              'datasetA4tHF1P.npz',
    "dataset":              'datasetFGHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     30,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})
#ran only for s = 3,10,15,20,30 + 5,8
for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]: #to avoid doubles with same number of measurements
    tmpConf = fgsfrun32t.copy({
        "path": f'fgsfrun32ts{s}',
        "dataset": f'datasetFGHF1Ps{s}.npz',
        "dimX": math.ceil(600/s),
    })
    exec(f"fgsfrun32ts{s} = tmpConf")

#================================== peak predictions, High Field data, y normalization, synth 10k dataset with distance correction, tune opt
# best after run: batchSize=8,dropout=0,hiddenDim=885,hiddenLayers=8,learningRate=0.0001,weightDecay=0
# best analysis : batchSize=8,dropout=0,hiddenDim=885,hiddenLayers=8,learningRate=0.0001,weightDecay=0
fgsfrun32ct = fgsfrun32t.copy({
    "path":                 'fgsfrun32ct',
    "dataset":              'datasetFGHF1Pc.npz',
})

#================================== peak predictions, High Field data, y normalization, synth 10k dataset with distance correction, tune opt, fixed SNR
# best after run: batchSize=16,dropout=0,hiddenDim=917,hiddenLayers=7,learningRate=0.0001,weightDecay=0
# best analysis : batchSize=16,dropout=0,hiddenDim=917,hiddenLayers=7,learningRate=0.0001,weightDecay=0, batchSize=16,dropout=0,hiddenDim=917,hiddenLayers=7,learningRate=0.0001,weightDecay=0
fgsfrun32ctN6 = fgsfrun32t.copy({
    "path":                 'fgsfrun32ctN6',
    "dataset":              'datasetFGHF1PcN6.npz',
})

# best after run:
# best analysis : batchSize=4,dropout=0,hiddenDim=446,hiddenLayers=7,learningRate=0.0001,weightDecay=0
fgsfrun32cct = fgsfrun32t.copy({
    "path":                 'fgsfrun32cct',
    "dataset":              'datasetFGHF1Pcc.npz',
})


#================================== peak predictions, High Field data, y normalization, synth 10k dataset with distance correction, different width tune opt
# best after run:
# best analysis :
fgsfrun32ctw15 = fgsfrun32t.copy({
    "path":                 'fgsfrun32ctw15',
    "dataset":              'datasetFGHF1Pcww15.npz',
})

# best after run:
# best analysis :
fgsfrun32ctw6 = fgsfrun32t.copy({
    "path":                 'fgsfrun32ctw6',
    "dataset":              'datasetFGHF1Pcww6.npz',
})

#================================== peak predictions, High Field data, y normalization, synth 10k dataset with distance correction, width in range tune opt
# best after run: batchSize=4,dropout=0,hiddenDim=870,hiddenLayers=6,learningRate=0.0001,weightDecay=0
# best analysis :
fgsfrun32ctwR = fgsfrun32t.copy({
    "path":                 'fgsfrun32ctwR',
    "dataset":              'datasetFGHF1PcwwR.npz',
})

#================================== peak predictions, High Field data, y normalization, synth 10k dataset with distance correction, width in range tune opt
# best after run: batchSize=8,dropout=0,hiddenDim=1018,hiddenLayers=8,learningRate=0.0001,weightDecay=0
# best analysis :
fgsfrun32ctwR2 = fgsfrun32t.copy({
    "path":                 'fgsfrun32ctwR2',
    "dataset":              'datasetFGHF1PcwwR2.npz',
})


#================================== (run by error, repetition of another setting) peak predictions, High Field data, y normalization, real and synth 1k dataset real test and valid, tune opt
# best after run: batchSize=16,dropout=0,hiddenDim=677,hiddenLayers=5,learningRate=0.0010,weightDecay=0
# best analysis :
fgsfrun33t = configGlobal.copy({
    "path":                 'fgsfrun33t',
    "dataset":              'datasetA4tHF1P.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     30,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#================================== peak predictions, High Field data, y normalization, synth 1k dataset, tune opt
# best after run: batchSize=2,dropout=0,hiddenDim=606,hiddenLayers=7,learningRate=0.0001,weightDecay=0
# best analysis : 
fgsfrun34t = configGlobal.copy({
    "path":                 'fgsfrun34t',
    "dataset":              'datasetFGHF1P1k.npz',
    "epochs":               100,
    "useTune":              True,
    "nonVerbose":           True,
    "tuneNumSamples":       1000,
    "tuneParallelProc":     20,
    "tuneGrace":            10,
    "tuneConf":             {
        "learningRate":         tune.choice([1e-2, 1e-3, 1e-4]),
        "dropout":              tune.choice([0, 0.2, 0.5]),
        "weightDecay":          tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batchSize":            tune.choice([2, 4, 8, 16, 32]),
        "hiddenLayers":         tune.lograndint(1, 32),
        "hiddenDim":            tune.lograndint(1, 1024),
    },
    "dimX":                 600,
    "dimY":                 8,
    "normalizeX":           True,
    "normalizeY":           True,
    "minMax":               [(2530, 3250)]*8,
    "finalActivation":      'sigmoid',
    "plotLossName":         "MSE",
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

#not run
for s in list(range(3,26))+[27,28,29,30,32,34,36,38,40]: #to avoid doubles with same number of measurements
    tmpConf = fgsfrun34t.copy({
        "path": f'fgsfrun34ts{s}',
        "dataset": f'datasetFGHF1P1ks{s}.npz',
    })
    exec(f"fgsfrun34ts{s} = tmpConf")


#================================== calculate predictions on real data
fgsfrun32tPred = fgsfrun32t.copy({
    "nonVerbose": False,
    "dataset":              'datasetFGHF1Prt.npz',
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})

fgsfrun34tPred = fgsfrun34t.copy({
    "nonVerbose": False,
    "dataset":              'datasetFGHF1Prt.npz',
    "modelLoad":            'best.pt',
    # "modelLoad":            'last.pt',
})



# ================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Use {} configName".format(sys.argv[0]))
    else:
        # conf = getattr(sys.modules['configurations'], sys.argv[1])
        conf = eval(format(sys.argv[1]))
        conf.print()
