import os
import numpy as np
import h5py
import argparse
import scipy.io

default = {
    'realTrainXKey': 'realX',
    'realTrainYKey': 'realY',
    'realValidXKey': 'realX',
    'realValidYKey': 'realY',
    'realTestXKey': 'realX',
    'realTestYKey': 'realY',
    'syntTrainXKey': 'trainX',
    'syntTrainYKey': 'trainY',
    'syntValidXKey': 'validX',
    'syntValidYKey': 'validY',
    'syntTestXKey': 'testX',
    'syntTestYKey': 'testY',
}

configurations = {
    'datasetAHF1P': { #Only real data
        **default,
        'saveFilename': 'datasets/datasetAHF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,50,1),
        'syntTrainSize': (0,0,1),
        'realValidSize': (50,93,1),
        'syntValidSize': (0,0,1),
        'realTestSize': (50,93,1),
        'syntTestSize': (0,0,1),
    },
    'datasetA2HF1P': { #Real data and 1000 synth
        **default,
        'saveFilename': 'datasets/datasetA2HF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,50,1),
        'syntTrainSize': (0,1000,1),
        'realValidSize': (50,93,1),
        'syntValidSize': (0,0,1),
        'realTestSize': (50,93,1),
        'syntTestSize': (0,0,1),
    },
    'datasetA2bHF1P': { #Real data and 10000 synth
        **default,
        'saveFilename': 'datasets/datasetA2bHF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,50,1),
        'syntTrainSize': (0,10000,1),
        'realValidSize': (50,93,1),
        'syntValidSize': (0,0,1),
        'realTestSize': (50,93,1),
        'syntTestSize': (0,0,1),
    },
    'datasetA3HF1P': { #Only real data, more shuffled (1 train, 1 valid/test, 1 train, 1 valid/test, ...)
        **default,
        'saveFilename': 'datasets/datasetA3HF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,None,2),
        'syntTrainSize': (0,0,1),
        'realValidSize': (1,None,2),
        'syntValidSize': (0,0,1),
        'realTestSize': (1,None,2),
        'syntTestSize': (0,0,1),
    },
    'datasetA3tHF1P': { #Only real data, more shuffled (1 train, 1 valid/test, 1 train, 1 valid/test, ...), different valid test
        **default,
        'saveFilename': 'datasets/datasetA3tHF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,None,3),
        'syntTrainSize': (0,0,1),
        'realValidSize': (1,None,3),
        'syntValidSize': (0,0,1),
        'realTestSize': (2,None,3),
        'syntTestSize': (0,0,1),
    },
    'datasetA4HF1P': { #Real data and 1000 synth, more shuffled (1 train, 1 valid/test, 1 train, 1 valid/test, ...)
        **default,
        'saveFilename': 'datasets/datasetA4HF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,None,2),
        'syntTrainSize': (0,1000,1),
        'realValidSize': (1,None,2),
        'syntValidSize': (0,0,1),
        'realTestSize': (1,None,2),
        'syntTestSize': (0,0,1),
    },
    'datasetA4tHF1P': { #Real data and 1000 synth, more shuffled (1 train, 1 valid/test, 1 train, 1 valid/test, ...), different valid test
        **default,
        'saveFilename': 'datasets/datasetA4tHF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,None,3),
        'syntTrainSize': (0,1000,1),
        'realValidSize': (1,None,3),
        'syntValidSize': (0,0,1),
        'realTestSize': (2,None,3),
        'syntTestSize': (0,0,1),
    },
    'datasetA4bHF1P': { #Real data and 10000 synth, more shuffled (1 train, 1 valid/test, 1 train, 1 valid/test, ...)
        **default,
        'saveFilename': 'datasets/datasetA4bHF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,None,2),
        'syntTrainSize': (0,10000,1),
        'realValidSize': (1,None,2),
        'syntValidSize': (0,0,1),
        'realTestSize': (1,None,2),
        'syntTestSize': (0,0,1),
    },
    'datasetA5HF1P': { #Only real data, random shuffled
        **default,
        'saveFilename': 'datasets/datasetA5HF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': "datasets/splitTrainRGHF1P.npy",
        'syntTrainSize': (0,0,1),
        'realValidSize': "datasets/splitTestRGHF1P.npy",
        'syntValidSize': (0,0,1),
        'realTestSize': "datasets/splitTestRGHF1P.npy",
        'syntTestSize': (0,0,1),
    },
    'datasetA6HF1P': { #Real data and 1000 synth, random shuffled
        **default,
        'saveFilename': 'datasets/datasetA6HF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': "datasets/splitTrainRGHF1P.npy",
        'syntTrainSize': (0,1000,1),
        'realValidSize': "datasets/splitTestRGHF1P.npy",
        'syntValidSize': (0,0,1),
        'realTestSize': "datasets/splitTestRGHF1P.npy",
        'syntTestSize': (0,0,1),
    },
    'datasetA7HF1P': { #Only synth data, same dimension of real split 2
        **default,
        'saveFilename': 'datasets/datasetA7HF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'realTrainSize': (0,0,1),
        'syntTrainSize': (0,93,2),
        'realValidSize': (0,0,1),
        'syntValidSize': (1,93,2),
        'realTestSize': (0,0,1),
        'syntTestSize': (1,93,2),
    },
    'datasetA7bHF1P': { #Only synth data, same dimension of real split 2, same test and valid
        **default,
        'saveFilename': 'datasets/datasetA7bHF1P.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFRGHF1P.npz',
        'syntTestXKey': 'validX',
        'syntTestYKey': 'validY',
        'realTrainSize': (0,0,1),
        'syntTrainSize': (0,93,2),
        'realValidSize': (0,0,1),
        'syntValidSize': (1,93,2),
        'realTestSize': (0,0,1),
        'syntTestSize': (1,93,2),
    },
    'datasetFGHF1P1k': { #Only synth data, 1k samples
        **default,
        'saveFilename': 'datasets/datasetFGHF1P1k.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFGHF1P.npz',
        'syntTestXKey': 'validX',
        'syntTestYKey': 'validY',
        'realTrainSize': (0,0,1),
        'syntTrainSize': (0,1000,1),
        'realValidSize': (0,0,1),
        'syntValidSize': (0,200,1),
        'realTestSize': (0,0,1),
        'syntTestSize': (0,200,1),
    },
    'datasetFGHF1Prt': { #Only real data, duplicated in train, valid and test
        **default,
        'saveFilename': 'datasets/datasetFGHF1Prt.npz',
        'realDataset': 'datasets/datasetRGHF1P.npz',
        'syntDataset': 'datasets/datasetFGHF1P.npz',
        'syntTestXKey': 'validX',
        'syntTestYKey': 'validY',
        'realTrainSize': (0,None,1),
        'syntTrainSize': (0,0,1),
        'realValidSize': (0,None,1),
        'syntValidSize': (0,0,1),
        'realTestSize': (0,None,1),
        'syntTestSize': (0,0,1),
    },
}

for i in ['', '2', '2b', '3', '4', '4b', '5', '6']:
    for s in range(2,41):
        configurations[f'datasetA{i}HF1Ps{s}'] = {
            **configurations[f'datasetA{i}HF1P'],
            'saveFilename': f'datasets/datasetA{i}HF1Ps{s}.npz',
            'realDataset': f'datasets/datasetRGHF1Ps{s}.npz',
            'syntDataset': f'datasets/datasetFRGHF1Ps{s}.npz',
        }

for s in range(2, 41):
    configurations[f'datasetFGHF1P1ks{s}'] = {
        **configurations[f'datasetFGHF1P1k'],
        'saveFilename': f'datasets/datasetFGHF1P1ks{s}.npz',
        'realDataset': f'datasets/datasetRGHF1Ps{s}.npz',
        'syntDataset': f'datasets/datasetFGHF1Ps{s}.npz',
    }

def main(confName):
    conf = configurations[confName]

    realData = np.load(conf['realDataset'])
    syntData = np.load(conf['syntDataset'])

    if type(conf['realTrainSize']) is tuple:
        realTrainSplit = np.arange(realData[conf['realTrainXKey']].shape[0])[conf['realTrainSize'][0]:conf['realTrainSize'][1]:conf['realTrainSize'][2]]
    elif type(conf['realTrainSize']) is str:
        realTrainSplit = np.load(conf['realTrainSize'])

    if type(conf['syntTrainSize']) is tuple:
        syntTrainSplit = np.arange(syntData[conf['syntTrainXKey']].shape[0])[conf['syntTrainSize'][0]:conf['syntTrainSize'][1]:conf['syntTrainSize'][2]]
    elif type(conf['syntTrainSize']) is str:
        syntTrainSplit = np.load(conf['syntTrainSize'])

    if type(conf['realValidSize']) is tuple:
        realValidSplit = np.arange(realData[conf['realValidXKey']].shape[0])[conf['realValidSize'][0]:conf['realValidSize'][1]:conf['realValidSize'][2]]
    elif type(conf['realValidSize']) is str:
        realValidSplit = np.load(conf['realValidSize'])

    if type(conf['syntValidSize']) is tuple:
        syntValidSplit = np.arange(syntData[conf['syntValidXKey']].shape[0])[conf['syntValidSize'][0]:conf['syntValidSize'][1]:conf['syntValidSize'][2]]
    elif type(conf['syntValidSize']) is str:
        syntValidSplit = np.load(conf['syntValidSize'])

    if type(conf['realTestSize']) is tuple:
        realTestSplit = np.arange(realData[conf['realTestXKey']].shape[0])[conf['realTestSize'][0]:conf['realTestSize'][1]:conf['realTestSize'][2]]
    elif type(conf['realTestSize']) is str:
        realTestSplit = np.load(conf['realTestSize'])

    if type(conf['syntTestSize']) is tuple:
        syntTestSplit = np.arange(syntData[conf['syntTestXKey']].shape[0])[conf['syntTestSize'][0]:conf['syntTestSize'][1]:conf['syntTestSize'][2]]
    elif type(conf['syntTestSize']) is str:
        syntTestSplit = np.load(conf['syntTestSize'])


    np.savez_compressed(
        conf['saveFilename'],
        freq = realData['realF'], #Take frequencies from real data, assume that for synthetic are equal
        trainX = np.concatenate((realData[conf['realTrainXKey']][realTrainSplit], syntData[conf['syntTrainXKey']][syntTrainSplit])),
        trainY = np.concatenate((realData[conf['realTrainYKey']][realTrainSplit], syntData[conf['syntTrainYKey']][syntTrainSplit])),
        validX = np.concatenate((realData[conf['realValidXKey']][realValidSplit], syntData[conf['syntValidXKey']][syntValidSplit])),
        validY = np.concatenate((realData[conf['realValidYKey']][realValidSplit], syntData[conf['syntValidYKey']][syntValidSplit])),
        testX = np.concatenate((realData[conf['realTestXKey']][realTestSplit], syntData[conf['syntTestXKey']][syntTestSplit])),
        testY = np.concatenate((realData[conf['realTestYKey']][realTestSplit], syntData[conf['syntTestYKey']][syntTestSplit])),
    )
    print(f"Saved in {conf['saveFilename']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
