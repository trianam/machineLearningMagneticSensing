import torch #to resolve strange bug with torch.load
import os
from pathlib import Path
import matlab.engine
import numpy as np
import configurations
import funPytorch as fun
import loaders
import argparse
import math
from matplotlib import pyplot as plt

thisConfigurations = {
    # ========== only synth
    'fgsfrun32tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'fullRef': {
                        'testData': f"datasets/datasetIA6HF1P.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32tPred",
                        'overwriteSub': 1,
                    },
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32tPred",
                        'overwriteSub': s,
                    } for s in [20,30]],
    },
}

def main(confName):
    conf = thisConfigurations[confName]

    refConf = conf['fullRef']
    refData = np.load(refConf['testData'])
    refTakeEvery = refConf['takeEvery']

    #Lorentzian fit
    eng = matlab.engine.start_matlab()
    # eng.addpath("~/matlab/toolbox/signal/signal")
    # eng.addpath("~/matlab/toolbox/stats/stats")
    # eng.addpath("~/matlab/toolbox/shared/statslib")
    eng.cd(r'sim', nargout=0)

    testSizeConf = conf['testSize']

    if 'testData' in conf:
        data = np.load(conf['testData'])
        if testSizeConf is None:
            testSize = data[f"{conf['testSet']}X"].shape[0]
        else:
            testSize = testSizeConf

    for iTp,tp in enumerate(conf['toPrint']):
        print(f"Set {iTp+1}/{len(conf['toPrint'])} - Calculating LF")

        if 'testData' in tp:
            data = np.load(tp['testData'])
            if testSizeConf is None:
                testSize = data[f"{conf['testSet']}X"].shape[0]
            else:
                testSize = testSizeConf

        takeEvery = tp['takeEvery']
        confName = tp['confName']

        iSample = 0
        found = 0
        foundLess = False
        while found < testSize:
            if conf['skipLF'] or iSample >= data[f"{conf['testSet']}X"].shape[0]:
                foundLess = True
                break

            f = matlab.double(data[conf['freqName']].reshape([1,-1])[:,::takeEvery].tolist())
            x = matlab.double(data[f"{conf['testSet']}X"][iSample].reshape([1,-1])[:, ::takeEvery].tolist())

            # for threshold in np.arange(0.00001, 0.01001, 0.00001):
            #     smooth_data, num_pks, initial_guess, locs = eng.getFitGuess(f,x,float(threshold), nargout=4)
            #     if num_pks == 8:
            #         break
            smooth_data, num_pks, initial_guess, locs = eng.getFitGuess(f,x,conf['minPeakHeight'],conf['minPeakProminence'],conf['smoothSpan'], nargout=4)

            if num_pks == 8:
                # yprime,params,resnorm,residual,conf = eng.lorentzian_fit_lf(f,x,2,2,8,initial_guess, nargout=5)
                _,params,_,_,confidence = eng.lorentzian_fit_lf(f,x,2,2,8,initial_guess, nargout=5)
                confidence = np.array(confidence)
                confidence = abs(confidence[:, 1] - confidence[:, 0])[:-1:3] #take only the avg peaks confidence
                p = np.array([params[0][i] for i in [0, 3, 6, 9, 12, 15, 18, 21]])
                y = data[f"{conf['testSet']}Y"][iSample]
                maes = abs(p - y).mean()

                found += 1

                xRef = refData[f"{conf['testSet']}X"][iSample].reshape([1, -1])[:, ::refTakeEvery]

                plt.plot(np.array(f)[0], xRef[0], 'C1')
                plt.plot(np.array(f)[0], np.array(x)[0], 'C0')
                plt.show()
                print("break")

            iSample+=1


    eng.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Configuration name')

    args = parser.parse_args()
    main(args.configName)