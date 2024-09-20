import numpy as np
import argparse
from matplotlib import pyplot as plt

configurations = {
    'datasetRGHF1P': { #real data
        'dataset': 'datasets/datasetRGHF1P.npz',
        'yKey': 'realY',
        'limit': None,
    },
    'datasetFRGHF1P': { #simulated
        'dataset': 'datasets/datasetFRGHF1P.npz',
        'yKey': 'trainY',
        'limit': (0,1000),
    },
    'datasetFGHF1P': { #simulated
        'dataset': 'datasets/datasetFGHF1P.npz',
        'yKey': 'trainY',
        'limit': None,
    },
    'datasetFGHF1Pc': { #simulated
        'dataset': 'datasets/datasetFGHF1Pc.npz',
        'yKey': 'trainY',
        'limit': None,
    },
    'datasetFGHF1Pcc': { #simulated
        'dataset': 'datasets/datasetFGHF1Pcc.npz',
        'yKey': 'trainY',
        'limit': None,
    },
}

def main(confName):
    conf = configurations[confName]

    data = np.load(conf['dataset'])
    peak = data[conf['yKey']]

    if not conf['limit'] is None:
        peak = peak[conf['limit'][0]:conf['limit'][1]]

    for p in range(peak.shape[1]):
        plt.hist(peak[:,p], alpha = 0.5)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
