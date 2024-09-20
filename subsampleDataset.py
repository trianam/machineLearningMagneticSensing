import argparse
import numpy as np


configurations = {}

for s in range(2, 41):
    configurations[f"datasetFGHF1Ps{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Ps{s}.npz',
        'originalDataset': "datasets/datasetFGHF1P.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pcs{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pcs{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pc.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pccs{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pccs{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pcc.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pw15s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pw15s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pw15.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pw10s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pw10s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pw10.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pw6s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pw6s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pw6.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pn10s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pn10s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pn10.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pn6s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pn6s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pn6.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pn2s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pn2s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pn2.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pmw15s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmw15s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmw15.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }
    configurations[f"datasetFGHF1Pmw10s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmw10s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmw10.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }
    configurations[f"datasetFGHF1Pmw6s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmw6s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmw6.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pmw15fns{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmw15fns{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmw15fn.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }
    configurations[f"datasetFGHF1Pmw10fns{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmw10fns{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmw10fn.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }
    configurations[f"datasetFGHF1Pmw6fns{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmw6fns{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmw6fn.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

    configurations[f"datasetFGHF1Pmn10s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmn10s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmn10.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }
    configurations[f"datasetFGHF1Pmn6s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmn6s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmn6.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }
    configurations[f"datasetFGHF1Pmn2s{s}"] = {
        'saveFilename': f'datasets/datasetFGHF1Pmn2s{s}.npz',
        'originalDataset': "datasets/datasetFGHF1Pmn2.npz",
        'subsampling': s,
        'subsamplingKeys': ['freq', 'trainX', 'validX', 'testX'],
    }

def main(confName):
    conf = configurations[confName]

    origData = np.load(conf['originalDataset'])
    subData = {}

    for k in origData.files:
        if k in conf['subsamplingKeys']:
            if len(origData[k].shape) == 1:
                subData[k] = origData[k][::conf['subsampling']]
            elif len(origData[k].shape) == 2:
                subData[k] = origData[k][:,::conf['subsampling']]
            else:
                raise Exception("Unconsistent dimension")
        else:
            subData[k] = origData[k]

    np.savez_compressed(conf['saveFilename'], **subData)
    print(f"Saved in {conf['saveFilename']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
