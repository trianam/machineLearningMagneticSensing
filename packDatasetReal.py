import os
import numpy as np
import h5py
import argparse
import scipy.io

configurations = {
    'datasetRGSF1': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetRGSF1.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1s2': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 2,
        'saveFilename': 'datasets/datasetRGSF1s2.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1s3': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 3,
        'saveFilename': 'datasets/datasetRGSF1s3.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1s4': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 4,
        'saveFilename': 'datasets/datasetRGSF1s4.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1s5': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 5,
        'saveFilename': 'datasets/datasetRGSF1s5.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1s6': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 6,
        'saveFilename': 'datasets/datasetRGSF1s6.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1s7': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 7,
        'saveFilename': 'datasets/datasetRGSF1s7.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1s8': {
        'y': [38.17, 25.5755, 67.3352], #'magnetic_field', 'B_theta', 'B_phi'
        'takeEvery': 8,
        'saveFilename': 'datasets/datasetRGSF1s8.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    # peak prediction
    'datasetRGSF1P': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetRGSF1P.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps2': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 2,
        'saveFilename': 'datasets/datasetRGSF1Ps2.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps3': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 3,
        'saveFilename': 'datasets/datasetRGSF1Ps3.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps4': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 4,
        'saveFilename': 'datasets/datasetRGSF1Ps4.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps5': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 5,
        'saveFilename': 'datasets/datasetRGSF1Ps5.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps6': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 6,
        'saveFilename': 'datasets/datasetRGSF1Ps6.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps7': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 7,
        'saveFilename': 'datasets/datasetRGSF1Ps7.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps8': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 8,
        'saveFilename': 'datasets/datasetRGSF1Ps8.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps9': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 9,
        'saveFilename': 'datasets/datasetRGSF1Ps9.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps10': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 10,
        'saveFilename': 'datasets/datasetRGSF1Ps10.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'datasetRGSF1Ps11': {
        'y': [2782.011278521792, 2804.127205748404, 2834.300604855577, 2855.585166324434, 2897.144553950816, 2917.062266268262, 2944.151221713331, 2962.930668995134], #peaks
        'takeEvery': 11,
        'saveFilename': 'datasets/datasetRGSF1Ps11.npz',
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
}

def main(confName):
    myConf = configurations[confName]

    if myConf['mat7.3']:
        meas = np.array(h5py.File(myConf['measFilename'], 'r').get(myConf['measDataname'])).reshape(-1)
        freq = np.array(h5py.File(myConf['freqFilename'], 'r').get(myConf['freqDataname'])).reshape(-1)
    else:
        meas = np.array(scipy.io.loadmat(myConf['measFilename']).get(myConf['measDataname'])).reshape(-1)
        freq = np.array(scipy.io.loadmat(myConf['freqFilename']).get(myConf['freqDataname'])).reshape(-1)

    if myConf['toReference']:
        if myConf['mat7.3']:
            ref = np.array(h5py.File(myConf['refFilename'], 'r').get(myConf['refDataname'])).reshape(-1)
        else:
            ref = np.array(scipy.io.loadmat(myConf['refFilename']).get(myConf['refDataname'])).reshape(-1)
        meas = (meas - ref) / np.average(ref)

    if myConf['invertPlot']:
        meas = -meas

    if not myConf['cutToFreqs'] is None:
        cutIndex = np.argwhere((freq >= myConf['cutToFreqs'][0]) & (freq <= myConf['cutToFreqs'][1])).reshape(-1)
        meas = meas[cutIndex]
        freq = freq[cutIndex]

    meas = meas.reshape((1,-1))[:,::myConf['takeEvery']]
    freq = freq.reshape((1,-1))[:,::myConf['takeEvery']]
    y = np.array(myConf['y']).reshape((1,-1))

    np.savez_compressed(myConf['saveFilename'], realF=freq, realX=meas, realY=y)
    print(f"Saved in {myConf['saveFilename']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
