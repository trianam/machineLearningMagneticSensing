import os
import numpy as np
import h5py
import argparse
import scipy.io
from matplotlib import pyplot as plt

configurations = {
    'datasetRGHF1P': {
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetRGHF1P.npz',
        'cutToFreqs': None,
        'invertPlot': False,
        'subsets': [f'{b}_{i}'  for b in ['x','x_2','x_3'] for i in range(1,12)] + [f'{b}_{i}'  for b in ['y','y_2','y_3'] for i in range(1,14)] + [f'z_{i}' for i in range(1,22)],
        'freqFilename': 'data/galya/high_field/freq.mat',
        'measFilename': 'data/galya/high_field/norm_{}.mat',
        'peakFilename': 'data/galya/high_field/peak_locations_norm_{}.mat',
        'noiseFilename': 'data/galya/high_field/noiseVec_{}.mat',
        'fieldFilename': 'data/galya/high_field/field_vals_{}.mat',
        'freqDataname': 'freq',
        'measDataname': 'norm',
        'peakDataname': 'peak_locations_norm_{}',
        'noiseDataname': 'noiseVec_{}',
        'fieldDataname': 'field_vals_{}',
    },
}
for s in range(2,41):
    configurations[f'datasetRGHF1Ps{s}'] = {
        'takeEvery': s,
        'saveFilename': f'datasets/datasetRGHF1Ps{s}.npz',
        'cutToFreqs': None,
        'invertPlot': False,
        'subsets': [f'{b}_{i}' for b in ['x', 'x_2', 'x_3'] for i in range(1, 12)] + [f'{b}_{i}' for b in
                                                                                      ['y', 'y_2', 'y_3'] for i in
                                                                                      range(1, 14)] + [f'z_{i}' for i in
                                                                                                       range(1, 22)],
        'freqFilename': 'data/galya/high_field/freq.mat',
        'measFilename': 'data/galya/high_field/norm_{}.mat',
        'peakFilename': 'data/galya/high_field/peak_locations_norm_{}.mat',
        'noiseFilename': 'data/galya/high_field/noiseVec_{}.mat',
        'fieldFilename': 'data/galya/high_field/field_vals_{}.mat',
        'freqDataname': 'freq',
        'measDataname': 'norm',
        'peakDataname': 'peak_locations_norm_{}',
        'noiseDataname': 'noiseVec_{}',
        'fieldDataname': 'field_vals_{}',
    }

def readMatFile(filename):
    if h5py.is_hdf5(filename):
        return h5py.File(filename, 'r')
    else:
        return scipy.io.loadmat(filename)

def main(confName):
    conf = configurations[confName]

    freq = np.array(readMatFile(conf['freqFilename']).get(conf['freqDataname'])).reshape(-1)

    if not conf['cutToFreqs'] is None:
        cutIndex = np.argwhere((freq >= conf['cutToFreqs'][0]) & (freq <= conf['cutToFreqs'][1])).reshape(-1)
        freq = freq[cutIndex]

    meas = []
    peak = []
    noise = []
    field = []
    for subset in conf['subsets']:
        subsetBase = "_".join(subset.split("_")[:-1])
        subsetIndex = int(subset.split("_")[-1]) -1

        meas.append(np.array(readMatFile(conf['measFilename'].format(subset)).get(conf['measDataname'])).reshape(-1))
        peak.append(np.array(readMatFile(conf['peakFilename'].format(subsetBase)).get(conf['peakDataname'].format(subsetBase)))[subsetIndex])
        noise.append(np.array(readMatFile(conf['noiseFilename'].format(subsetBase)).get(conf['noiseDataname'].format(subsetBase)))[0,subsetIndex])
        field.append(np.array(readMatFile(conf['fieldFilename'].format(subsetBase)).get(conf['fieldDataname'].format(subsetBase)))[subsetIndex])

        if conf['invertPlot']:
            meas[-1] = -meas[-1]

        if not conf['cutToFreqs'] is None:
            meas[-1] = meas[-1][cutIndex]

    # freq = freq[::conf['takeEvery']]
    # meas = np.stack(meas)[:,::conf['takeEvery']]
    peak = np.stack(peak)
    # noise = np.stack(noise)
    # field = np.stack(field)

    # np.savez_compressed(conf['saveFilename'], realF=freq, realX=meas, realY=peak, realN=noise, realM=field)

    for p in range(peak.shape[1]):
        plt.hist(peak[:,p])

    plt.show()


    # print(f"Saved in {conf['saveFilename']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
