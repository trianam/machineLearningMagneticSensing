import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch

import configurations
import sys
import funPytorch as fun
from torch.utils.data import TensorDataset, DataLoader


datasets = {
    'data1': {
        'toReference': True,
        'mat7.3': True,
        'measFilename': 'data/galya/1/ESR_meas_100_simu_freqs_3.mat',
        'refFilename': 'data/galya/1/ESR_ref_100_simu_freqs_3.mat',
        'freqFilename': 'data/galya/1/freq.mat',
        'groundTruth': [98, 24.99, 13.3742],
        'measDataname': 'averaged_meas',
        'refDataname': 'averaged_ref',
        'freqDataname': 'freq',
        'modelToUse': 'fgrun1',
    },
}

def main(confName):
    data = datasets[confName]

    if data['mat7.3']:
        meas = np.array(h5py.File(data['measFilename'], 'r').get(data['measDataname'])).reshape(-1)
        freq = np.array(h5py.File(data['freqFilename'], 'r').get(data['freqDataname'])).reshape(-1)
    else:
        meas = np.array(scipy.io.loadmat(data['measFilename']).get(data['measDataname'])).reshape(-1)
        freq = np.array(scipy.io.loadmat(data['freqFilename']).get(data['freqDataname'])).reshape(-1)


    if data['toReference']:
        if data['mat7.3']:
            ref = np.array(h5py.File(data['refFilename'], 'r').get(data['refDataname'])).reshape(-1)
        else:
            ref = np.array(scipy.io.loadmat(data['refFilename']).get(data['refDataname'])).reshape(-1)
        meas = (meas-ref)/np.average(ref)

    meas = -meas

    conf = eval('configurations.{}'.format(data['modelToUse']))
    device = "cuda:0"
    print("======= LOAD MODEL")
    model, optim, loadEpoch, _ = fun.loadModel(conf, device)
    print("======= LOAD DATA")
    dataset = TensorDataset(torch.Tensor(meas.reshape(1,-1)), torch.Tensor([data['groundTruth']]))  # create your datset
    dataloaders = {'valid': DataLoader(dataset, batch_size=conf.batchSize)}
    print("======= CALCULATE PREDICTIONS")
    preds = fun.predict(conf, model, dataloaders, loadEpoch, toSave=False, toReturn=True)

    print(f"Predicted {preds}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)