import os
import numpy as np
import h5py
import argparse
from rich.progress import track
import matlab.engine


configurations = {
    'datasetFGHF1Pm': {
        'simScript': 'sim/mock_diamond2_new_galya.m',
        'N': 600,
        'center_freq': 2890,
        'width': 10,
        'half_window_size': 360,
        'noiseSigma': (0.0002, 0.001), # 0.0002725640441769875 - 0.0009385689629442084
        'B_mag': (80, 120), # 808.079662936 - 1135.134239139
        'B_theta': (57, 73), # 57.65151589 - 72.01018995
        'B_phi': (54, 80), # 54.01828013 - 79.26266928
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'trainSize': 1000,
        'validSize': 1000,
        'testSize': 1000,
        'validFrom': None,
        'takeEvery': 1,
        'minPeakDist': 5,
        'saveFilename': 'datasets/datasetFGHF1Pm.npz'
    },
}

configurations['datasetFGHF1Pmw'] = {
    **configurations['datasetFGHF1Pm'],
    'multipleConf': [
        {
            'saveFilename': 'datasets/datasetFGHF1Pmw15.npz',
            'key': 'width',
            'value': 15,
        },
        {
            'saveFilename': 'datasets/datasetFGHF1Pmw10.npz',
            'key': 'width',
            'value': 10,
        },
        {
            'saveFilename': 'datasets/datasetFGHF1Pmw6.npz',
            'key': 'width',
            'value': 6,
        },
    ],
}

configurations['datasetFGHF1Pmwfn'] = {
    **configurations['datasetFGHF1Pm'],
    'noiseSigma': 0.0006,
    'multipleConf': [
        {
            'saveFilename': 'datasets/datasetFGHF1Pmw15fn.npz',
            'key': 'width',
            'value': 15,
        },
        {
            'saveFilename': 'datasets/datasetFGHF1Pmw10fn.npz',
            'key': 'width',
            'value': 10,
        },
        {
            'saveFilename': 'datasets/datasetFGHF1Pmw6fn.npz',
            'key': 'width',
            'value': 6,
        },
    ],
}


configurations['datasetFGHF1Pmn'] = {
    **configurations['datasetFGHF1Pm'],
    'multipleConf': [
        {
            'saveFilename': 'datasets/datasetFGHF1Pmn10.npz',
            'key': 'noiseSigma',
            'value': 0.001,
        },
        {
            'saveFilename': 'datasets/datasetFGHF1Pmn6.npz',
            'key': 'noiseSigma',
            'value': 0.0006,
        },
        {
            'saveFilename': 'datasets/datasetFGHF1Pmn2.npz',
            'key': 'noiseSigma',
            'value': 0.0002,
        },
    ],
}


def main(confName):
    conf = configurations[confName]
    workspace = os.path.dirname(conf['simScript'])
    funName = os.path.splitext(os.path.basename(conf['simScript']))[0]

    eng = matlab.engine.start_matlab()
    simPath = eng.genpath(workspace)
    eng.addpath(simPath, nargout=0)

    N = conf['N']
    center_freq = conf['center_freq']
    half_window_size = conf['half_window_size']

    freq = None
    trainX = []
    trainY = []
    validX = []
    validY = []
    testX = []
    testY = []
    
    for iMult in range(len(conf['multipleConf'])):
        trainX.append([])
        trainY.append([])
        validX.append([])
        validY.append([])
        testX.append([])
        testY.append([])

    for dataSplit, dataSize, listX, listY in [('train', conf['trainSize'], trainX, trainY), ('valid', conf['validSize'], validX, validY), ('test', conf['testSize'], testX, testY)]:
        for _ in track(range(dataSize), f"Generating {dataSplit} samples"):
            for iMult, mult in enumerate(conf['multipleConf']):
                if mult['key'] == 'noiseSigma':
                    noiseSigmaValue = mult['value']
                else:
                    noiseSigmaValue = conf['noiseSigma']

                if mult['key'] == 'width':
                    widthValue = mult['value']
                else:
                    widthValue = conf['width']

                if type(noiseSigmaValue) is list or type(noiseSigmaValue) is tuple:
                    noiseSigma = np.random.uniform(noiseSigmaValue[0], noiseSigmaValue[1])
                else:
                    noiseSigma = noiseSigmaValue
                B_mag = np.random.uniform(conf['B_mag'][0], conf['B_mag'][1])
                B_theta = np.random.uniform(conf['B_theta'][0], conf['B_theta'][1])
                B_phi = np.random.uniform(conf['B_phi'][0], conf['B_phi'][1])
                if type(widthValue) is list or type(widthValue) is tuple:
                    width = np.random.uniform(widthValue[0], widthValue[1])
                else:
                    width = widthValue
                diamond = eval(
                    f"eng.{funName}(float(N), float(center_freq), float(width), float(half_window_size), float(noiseSigma), float(B_mag), float(B_theta), float(B_phi))")
                eng.workspace["wDiamond"] = diamond
                if conf['minPeakDist'] is not None:
                    peaks = np.array(eng.eval(f"wDiamond.{conf['yDataName']}")).reshape(-1)
                    while not (np.array([peaks[i+1]-peaks[i] for i in range(len(peaks)-1)])>conf['minPeakDist']).all():
                        B_mag = np.random.uniform(conf['B_mag'][0], conf['B_mag'][1])
                        B_theta = np.random.uniform(conf['B_theta'][0], conf['B_theta'][1])
                        B_phi = np.random.uniform(conf['B_phi'][0], conf['B_phi'][1])
                        diamond = eval(
                            f"eng.{funName}(float(N), float(center_freq), float(width), float(half_window_size), float(noiseSigma), float(B_mag), float(B_theta), float(B_phi))")
                        eng.workspace["wDiamond"] = diamond

                        peaks = np.array(eng.eval(f"wDiamond.{conf['yDataName']}")).reshape(-1)

                if freq is None:
                    freq = np.array(eng.eval(f"wDiamond.{conf['fDataName']}")).reshape(-1)[::conf['takeEvery']]
                listX[iMult].append(np.array(eng.eval(f"wDiamond.{conf['xDataName']}")).reshape(-1)[::conf['takeEvery']])
                listY[iMult].append(np.array(eng.eval(f"wDiamond.{conf['yDataName']}")).reshape(-1))

        if dataSize == 0:
            for iMult in range(len(conf['multipleConf'])):
                listX[iMult].append(np.array([]))
                listY[iMult].append(np.array([]))

    for iMult in range(len(conf['multipleConf'])):
        trainX[iMult] = np.stack(trainX[iMult])
        trainY[iMult] = np.stack(trainY[iMult])
        validX[iMult] = np.stack(validX[iMult])
        validY[iMult] = np.stack(validY[iMult])
        testX[iMult] = np.stack(testX[iMult])
        testY[iMult] = np.stack(testY[iMult])

    if not conf['validFrom'] is None:
        otherDataset = np.load(conf['validFrom'])
        for iMult in range(len(conf['multipleConf'])):
            validX[iMult] = otherDataset['realX']
            validY[iMult] = otherDataset['realY']

    for iMult,saveFilename in enumerate([m['saveFilename'] for m in conf['multipleConf']]):
        np.savez_compressed(saveFilename, freq=freq, trainX=trainX[iMult], trainY=trainY[iMult], validX=validX[iMult], validY=validY[iMult], testX=testX[iMult], testY=testY[iMult])
        print(f"Saved in {saveFilename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
