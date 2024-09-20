import numpy as np
import argparse
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True

configurations = {
    'dataset1': {
        'filename': 'datasets/dataset1.npz',
        'freqName': 'trainF',
        'xName': 'trainX',
        'yName': 'trainY',
        'sample': 0,
    },
    'dataset2': {
        'filename': 'datasets/dataset2.npz',
        'freqName': 'trainF',
        'xName': 'trainX',
        'yName': 'trainY',
        'sample': 0,
    },
    'dataset4': {
        'filename': 'datasets/dataset4.npz',
        'freqName': 'trainF',
        'xName': 'trainX',
        'yName': 'trainY',
        'sample': 0,
    },
    'datasetG1': {
        'filename': 'datasets/datasetG1.npz',
        'freqName': 'trainF',
        'xName': 'trainX',
        'yName': 'trainY',
        'sample': 0,
    },
    'datasetG2': {
        'filename': 'datasets/datasetG2.npz',
        'freqName': 'trainF',
        'xName': 'trainX',
        'yName': 'trainY',
        'sample': 0,
    },
    'datasetG3': {
        'filename': 'datasets/datasetG3.npz',
        'freqName': 'trainF',
        'xName': 'trainX',
        'yName': 'trainY',
        'sample': 0,
    },
    'datasetG4': {
        'filename': 'datasets/datasetG4.npz',
        'freqName': 'trainF',
        'xName': 'trainX',
        'yName': 'trainY',
        'sample': 0,
    },
    'RGHF1P': {
        'filename': 'datasets/datasetRGHF1P.npz',
        'freqName': 'realF',
        'xName': 'realX',
        'yName': 'realY',
        'sample': 1,
    },
    'FRGHF1P': {
        'filename': 'datasets/datasetFRGHF1P.npz',
        'freqName': 'freq',
        'xName': 'testX',
        'yName': 'testY',
        'sample': 1,
    },
    'FGHF1Pc': {
        'filename': 'datasets/datasetFGHF1Pc.npz',
        'freqName': 'freq',
        'xName': 'testX',
        'yName': 'testY',
        'sample': 1,
    },
}

def main(confName):
    myConf = configurations[confName]
    dataset = np.load(myConf['filename'])

    freq = dataset[myConf['freqName']]
    if len(freq.shape)>1:
        freq = freq[0]
    x = dataset[myConf['xName']][myConf['sample']]
    y = dataset[myConf['yName']][myConf['sample']]

    plt.plot(freq, x)
    plt.title(r"Dataset {}; $B$ = {}".format(confName, y))
    plt.savefig('img/dataPlot.pdf')
    plt.show()

    #plt.hist(dataset['trainY'])
    #plt.title("Mag field freq")
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
