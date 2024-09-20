import os
import numpy as np
import h5py
import argparse
from rich.progress import track
import matlab.engine


configurations = {
    'datasetFGHF1P': {
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
        'trainSize': 10000,
        'validSize': 2000,
        'testSize': 2000,
        'validFrom': None,
        'takeEvery': 1,
        'minPeakDist': None,
        'saveFilename': 'datasets/datasetFGHF1P.npz'
    },
    'datasetFRGHF1P': {
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
        'trainSize': 10000,
        'validSize': 0,
        'testSize': 2000,
        'validFrom': "datasets/datasetRGHF1P.npz",
        'takeEvery': 1,
        'minPeakDist': None,
        'saveFilename': 'datasets/datasetFRGHF1P.npz'
    },
}

for s in range(2,41):
    configurations[f'datasetFRGHF1Ps{s}'] = {
        'simScript': 'sim/mock_diamond2_new_galya.m',
        'N': 600,
        'center_freq': 2890,
        'width': 10,
        'half_window_size': 360,
        'noiseSigma': (0.0002, 0.001),  # 0.0002725640441769875 - 0.0009385689629442084
        'B_mag': (80, 120),  # 808.079662936 - 1135.134239139
        'B_theta': (57, 73),  # 57.65151589 - 72.01018995
        'B_phi': (54, 80),  # 54.01828013 - 79.26266928
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'trainSize': 10000,
        'validSize': 0,
        'testSize': 2000,
        'validFrom': f"datasets/datasetRGHF1Ps{s}.npz",
        'takeEvery': s,
        'minPeakDist': None,
        'saveFilename': f'datasets/datasetFRGHF1Ps{s}.npz'
    }

configurations['datasetFGHF1Pc'] = {
    **configurations['datasetFGHF1P'],
    'minPeakDist': 5,
    'saveFilename': 'datasets/datasetFGHF1Pc.npz'
}

configurations['datasetFGHF1PcN6'] = {
    **configurations['datasetFGHF1Pc'],
    'noiseSigma': 0.0006,
    'saveFilename': 'datasets/datasetFGHF1PcN6.npz'
}

configurations['datasetFGHF1Pcc'] = {
    **configurations['datasetFGHF1P'],
    'minPeakDist': 50,
    'saveFilename': 'datasets/datasetFGHF1Pcc.npz'
}

configurations['datasetFGHF1Pcww15'] = {
    **configurations['datasetFGHF1P'],
    'minPeakDist': 5,
    'width': 15,
    'saveFilename': 'datasets/datasetFGHF1Pcww15.npz'
}

configurations['datasetFGHF1Pcww6'] = {
    **configurations['datasetFGHF1P'],
    'minPeakDist': 5,
    'width': 6,
    'saveFilename': 'datasets/datasetFGHF1Pcww6.npz'
}

configurations['datasetFGHF1PcwwR'] = {
    **configurations['datasetFGHF1P'],
    'minPeakDist': 5,
    'width': (5,16),
    'saveFilename': 'datasets/datasetFGHF1PcwwR.npz'
}

configurations['datasetFGHF1PcwwR2'] = {
    **configurations['datasetFGHF1P'],
    'minPeakDist': 5,
    'width': (3,18),
    'saveFilename': 'datasets/datasetFGHF1PcwwR2.npz'
}

configurations['datasetFGHF1Pw15'] = {
    **configurations['datasetFGHF1P'],
    'saveFilename': 'datasets/datasetFGHF1Pw15.npz',
    'minPeakDist': 5,
    'width': 15,
    'trainSize': 100,
    'validSize': 100,
    'testSize': 100,
}

configurations['datasetFGHF1Pw10'] = {
    **configurations['datasetFGHF1P'],
    'saveFilename': 'datasets/datasetFGHF1Pw10.npz',
    'minPeakDist': 5,
    'width': 10,
    'trainSize': 100,
    'validSize': 100,
    'testSize': 100,
}

configurations['datasetFGHF1Pw6'] = {
    **configurations['datasetFGHF1P'],
    'saveFilename': 'datasets/datasetFGHF1Pw6.npz',
    'minPeakDist': 5,
    'width': 6,
    'trainSize': 100,
    'validSize': 100,
    'testSize': 100,
}

configurations['datasetFGHF1Pn10'] = {
    **configurations['datasetFGHF1P'],
    'saveFilename': 'datasets/datasetFGHF1Pn10.npz',
    'minPeakDist': 5,
    'noiseSigma': 0.001,
    'trainSize': 100,
    'validSize': 100,
    'testSize': 100,
}

configurations['datasetFGHF1Pn6'] = {
    **configurations['datasetFGHF1P'],
    'saveFilename': 'datasets/datasetFGHF1Pn6.npz',
    'minPeakDist': 5,
    'noiseSigma': 0.0006,
    'trainSize': 100,
    'validSize': 100,
    'testSize': 100,
}

configurations['datasetFGHF1Pn2'] = {
    **configurations['datasetFGHF1P'],
    'saveFilename': 'datasets/datasetFGHF1Pn2.npz',
    'minPeakDist': 5,
    'noiseSigma': 0.0002,
    'trainSize': 100,
    'validSize': 100,
    'testSize': 100,
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

    for dataSplit, dataSize, listX, listY in [('train', conf['trainSize'], trainX, trainY), ('valid', conf['validSize'], validX, validY), ('test', conf['testSize'], testX, testY)]:
        for _ in track(range(dataSize), f"Generating {dataSplit} samples"):
            if type(conf['noiseSigma']) is list or type(conf['noiseSigma']) is tuple:
                noiseSigma = np.random.uniform(conf['noiseSigma'][0], conf['noiseSigma'][1])
            else:
                noiseSigma = conf['noiseSigma']
            B_mag = np.random.uniform(conf['B_mag'][0], conf['B_mag'][1])
            B_theta = np.random.uniform(conf['B_theta'][0], conf['B_theta'][1])
            B_phi = np.random.uniform(conf['B_phi'][0], conf['B_phi'][1])
            if type(conf['width']) is list or type(conf['width']) is tuple:
                width = np.random.uniform(conf['width'][0], conf['width'][1])
            else:
                width = conf['width']
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
            listX.append(np.array(eng.eval(f"wDiamond.{conf['xDataName']}")).reshape(-1)[::conf['takeEvery']])
            listY.append(np.array(eng.eval(f"wDiamond.{conf['yDataName']}")).reshape(-1))

        if dataSize == 0:
            listX.append(np.array([]))
            listY.append(np.array([]))

    trainX = np.stack(trainX)
    trainY = np.stack(trainY)
    validX = np.stack(validX)
    validY = np.stack(validY)
    testX = np.stack(testX)
    testY = np.stack(testY)

    if not conf['validFrom'] is None:
        otherDataset = np.load(conf['validFrom'])
        validX = otherDataset['realX']
        validY = otherDataset['realY']

    np.savez_compressed(conf['saveFilename'], freq=freq, trainX=trainX, trainY=trainY, validX=validX, validY=validY, testX=testX, testY=testY)
    print(f"Saved in {conf['saveFilename']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
