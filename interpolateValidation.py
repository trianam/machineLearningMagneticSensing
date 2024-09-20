import numpy as np
import argparse
import scipy.interpolate

# those are used only as a base for the for after
configurations = {
    'datasetIAHF1P': { #Only real data
        'saveFilename': 'datasets/datasetIAHF1P.npz',
        'trainDataset': 'datasets/datasetAHF1P.npz',
        'validDataset': 'datasets/datasetAHF1P.npz',
        'type': 'linear',
    },
    'datasetIA2HF1P': { #Real data and 1000 synth
        'saveFilename': 'datasets/datasetIA2HF1P.npz',
        'trainDataset': 'datasets/datasetA2HF1P.npz',
        'validDataset': 'datasets/datasetA2HF1P.npz',
        'type': 'linear',
    },
    'datasetIA3HF1P': { #Only real data, more shuffled (1 train, 1 valid/test, 1 train, 1 valid/test, ...)
        'saveFilename': 'datasets/datasetIA3HF1P.npz',
        'trainDataset': 'datasets/datasetA3HF1P.npz',
        'validDataset': 'datasets/datasetA3HF1P.npz',
        'type': 'linear',
    },
    'datasetIA4HF1P': { #Real data and 1000 synth, more shuffled (1 train, 1 valid/test, 1 train, 1 valid/test, ...)
        'saveFilename': 'datasets/datasetIA4HF1P.npz',
        'trainDataset': 'datasets/datasetA4HF1P.npz',
        'validDataset': 'datasets/datasetA4HF1P.npz',
        'type': 'linear',
    },

    'datasetIA5HF1P': { #Only synth 1k
        'saveFilename': 'datasets/datasetIA5HF1P.npz',
        'trainDataset': 'datasets/datasetFGHF1P1k.npz',
        'validDataset': 'datasets/datasetFGHF1P1k.npz',
        'type': 'linear',
    },
    'datasetIA6HF1P': { #Only synth 10k
        'saveFilename': 'datasets/datasetIA6HF1P.npz',
        'trainDataset': 'datasets/datasetFGHF1P.npz',
        'validDataset': 'datasets/datasetFGHF1P.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pc': { #Only synth 10k, peaks distance check
        'saveFilename': 'datasets/datasetIA6HF1Pc.npz',
        'trainDataset': 'datasets/datasetFGHF1Pc.npz',
        'validDataset': 'datasets/datasetFGHF1Pc.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pcc': { #Only synth 10k, peaks distance check more
        'saveFilename': 'datasets/datasetIA6HF1Pcc.npz',
        'trainDataset': 'datasets/datasetFGHF1Pcc.npz',
        'validDataset': 'datasets/datasetFGHF1Pcc.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pw15': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pw15.npz',
        'trainDataset': 'datasets/datasetFGHF1Pw15.npz',
        'validDataset': 'datasets/datasetFGHF1Pw15.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pw10': {  # Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pw10.npz',
        'trainDataset': 'datasets/datasetFGHF1Pw10.npz',
        'validDataset': 'datasets/datasetFGHF1Pw10.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pw6': {  # Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pw6.npz',
        'trainDataset': 'datasets/datasetFGHF1Pw6.npz',
        'validDataset': 'datasets/datasetFGHF1Pw6.npz',
        'type': 'linear',
    },

    'datasetIA6HF1Pn10': {  # Only synth 100 test, defined noise
        'saveFilename': 'datasets/datasetIA6HF1Pn10.npz',
        'trainDataset': 'datasets/datasetFGHF1Pn10.npz',
        'validDataset': 'datasets/datasetFGHF1Pn10.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pn6': {  # Only synth 100 test, defined noise
        'saveFilename': 'datasets/datasetIA6HF1Pn6.npz',
        'trainDataset': 'datasets/datasetFGHF1Pn6.npz',
        'validDataset': 'datasets/datasetFGHF1Pn6.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pn2': {  # Only synth 100 test, defined noise
        'saveFilename': 'datasets/datasetIA6HF1Pn2.npz',
        'trainDataset': 'datasets/datasetFGHF1Pn2.npz',
        'validDataset': 'datasets/datasetFGHF1Pn2.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmw15': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmw15.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmw15.npz',
        'validDataset': 'datasets/datasetFGHF1Pmw15.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmw10': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmw10.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmw10.npz',
        'validDataset': 'datasets/datasetFGHF1Pmw10.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmw6': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmw6.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmw6.npz',
        'validDataset': 'datasets/datasetFGHF1Pmw6.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmw15fn': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmw15fn.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmw15fn.npz',
        'validDataset': 'datasets/datasetFGHF1Pmw15fn.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmw10fn': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmw10fn.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmw10fn.npz',
        'validDataset': 'datasets/datasetFGHF1Pmw10fn.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmw6fn': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmw6fn.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmw6fn.npz',
        'validDataset': 'datasets/datasetFGHF1Pmw6fn.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmn10': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmn10.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmn10.npz',
        'validDataset': 'datasets/datasetFGHF1Pmn10.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmn6': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmn6.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmn6.npz',
        'validDataset': 'datasets/datasetFGHF1Pmn6.npz',
        'type': 'linear',
    },
    'datasetIA6HF1Pmn2': { #Only synth 100 test, defined width
        'saveFilename': 'datasets/datasetIA6HF1Pmn2.npz',
        'trainDataset': 'datasets/datasetFGHF1Pmn2.npz',
        'validDataset': 'datasets/datasetFGHF1Pmn2.npz',
        'type': 'linear',
    },
}

for i in ['', '2', '3', '4']:
    for s in range(2,41):
        configurations[f'datasetIA{i}HF1Ps{s}'] = {
            **configurations[f'datasetIA{i}HF1P'],
            'saveFilename': f'datasets/datasetIA{i}HF1Ps{s}.npz',
            'validDataset': f'datasets/datasetA{i}HF1Ps{s}.npz',
        }

for s in range(2,41):
    configurations[f'datasetIA5HF1Ps{s}'] = {
        **configurations[f'datasetIA5HF1P'],
        'saveFilename': f'datasets/datasetIA5HF1Ps{s}.npz',
        'validDataset': f'datasets/datasetFGHF1P1ks{s}.npz',
    }

    configurations[f'datasetIA6HF1Ps{s}'] = {
        **configurations[f'datasetIA6HF1P'],
        'saveFilename': f'datasets/datasetIA6HF1Ps{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Ps{s}.npz',
    }

    configurations[f'datasetIA6HF1Pcs{s}'] = {
        **configurations[f'datasetIA6HF1Pc'],
        'saveFilename': f'datasets/datasetIA6HF1Pcs{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pcs{s}.npz',
    }

    configurations[f'datasetIA6HF1Pccs{s}'] = {
        **configurations[f'datasetIA6HF1Pcc'],
        'saveFilename': f'datasets/datasetIA6HF1Pccs{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pccs{s}.npz',
    }

for s in range(2,11):
    configurations[f'datasetIA6HF1Pw15s{s}'] = {
        **configurations[f'datasetIA6HF1Pw15'],
        'saveFilename': f'datasets/datasetIA6HF1Pw15s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pw15s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pw10s{s}'] = {
        **configurations[f'datasetIA6HF1Pw10'],
        'saveFilename': f'datasets/datasetIA6HF1Pw10s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pw10s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pw6s{s}'] = {
        **configurations[f'datasetIA6HF1Pw6'],
        'saveFilename': f'datasets/datasetIA6HF1Pw6s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pw6s{s}.npz',
    }

    configurations[f'datasetIA6HF1Pn10s{s}'] = {
        **configurations[f'datasetIA6HF1Pn10'],
        'saveFilename': f'datasets/datasetIA6HF1Pn10s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pn10s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pn6s{s}'] = {
        **configurations[f'datasetIA6HF1Pn6'],
        'saveFilename': f'datasets/datasetIA6HF1Pn6s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pn6s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pn2s{s}'] = {
        **configurations[f'datasetIA6HF1Pn2'],
        'saveFilename': f'datasets/datasetIA6HF1Pn2s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pn2s{s}.npz',
    }

    configurations[f'datasetIA6HF1Pmw15s{s}'] = {
        **configurations[f'datasetIA6HF1Pmw15'],
        'saveFilename': f'datasets/datasetIA6HF1Pmw15s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmw15s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pmw10s{s}'] = {
        **configurations[f'datasetIA6HF1Pmw10'],
        'saveFilename': f'datasets/datasetIA6HF1Pmw10s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmw10s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pmw6s{s}'] = {
        **configurations[f'datasetIA6HF1Pmw6'],
        'saveFilename': f'datasets/datasetIA6HF1Pmw6s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmw6s{s}.npz',
    }

    configurations[f'datasetIA6HF1Pmw15fns{s}'] = {
        **configurations[f'datasetIA6HF1Pmw15fn'],
        'saveFilename': f'datasets/datasetIA6HF1Pmw15fns{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmw15fns{s}.npz',
    }
    configurations[f'datasetIA6HF1Pmw10fns{s}'] = {
        **configurations[f'datasetIA6HF1Pmw10fn'],
        'saveFilename': f'datasets/datasetIA6HF1Pmw10fns{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmw10fns{s}.npz',
    }
    configurations[f'datasetIA6HF1Pmw6fns{s}'] = {
        **configurations[f'datasetIA6HF1Pmw6fn'],
        'saveFilename': f'datasets/datasetIA6HF1Pmw6fns{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmw6fns{s}.npz',
    }

    configurations[f'datasetIA6HF1Pmn10s{s}'] = {
        **configurations[f'datasetIA6HF1Pmn10'],
        'saveFilename': f'datasets/datasetIA6HF1Pmn10s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmn10s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pmn6s{s}'] = {
        **configurations[f'datasetIA6HF1Pmn6'],
        'saveFilename': f'datasets/datasetIA6HF1Pmn6s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmn6s{s}.npz',
    }
    configurations[f'datasetIA6HF1Pmn2s{s}'] = {
        **configurations[f'datasetIA6HF1Pmn2'],
        'saveFilename': f'datasets/datasetIA6HF1Pmn2s{s}.npz',
        'validDataset': f'datasets/datasetFGHF1Pmn2s{s}.npz',
    }


def main(confName):
    conf = configurations[confName]

    trainData = np.load(conf['trainDataset'])
    validData = np.load(conf['validDataset'])

    validInterp = scipy.interpolate.interp1d(validData['freq'], validData['validX'], kind=conf['type'], fill_value='extrapolate')
    testInterp = scipy.interpolate.interp1d(validData['freq'], validData['testX'], kind=conf['type'], fill_value='extrapolate') #assume test and valid from same dataset

    np.savez_compressed(
        conf['saveFilename'],
        freq = trainData['freq'],
        trainX = trainData['trainX'],
        trainY = trainData['trainY'],
        validX = validInterp(trainData['freq']),
        validY = validData['validY'],
        testX = testInterp(trainData['freq']),
        testY = validData['testY'],
    )
    print(f"Saved in {conf['saveFilename']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
