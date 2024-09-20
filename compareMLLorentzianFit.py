import os
import matlab.engine
import h5py
import numpy as np
import configurations
import funPytorch as fun
import loaders

testSize = 100

testPath = "data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test"

toPrint = [
    {
        'takeEvery': 1,
        'confName': "fgsfrun17",
    },
    {
        'takeEvery': 2,
        'confName': "fgsfrun17s2",
    },
    {
        'takeEvery': 3,
        'confName': "fgsfrun17s3",
    },

    {
        'takeEvery': 4,
        'confName': "fgsfrun17s4",
    },

    {
        'takeEvery': 5,
        'confName': "fgsfrun17s5",
    },

    {
        'takeEvery': 6,
        'confName': "fgsfrun17s6",
    },

    {
        'takeEvery': 7,
        'confName': "fgsfrun17s7",
    },

    {
        'takeEvery': 8,
        'confName': "fgsfrun17s8",
    },

    {
        'takeEvery': 9,
        'confName': "fgsfrun17s9",
    },
    {
        'takeEvery': 10,
        'confName': "fgsfrun17s10",
    },
    {
        'takeEvery': 11,
        'confName': "fgsfrun17s11",
    },
    {
        'takeEvery': 12,
        'confName': "fgsfrun17s12",
    },
    {
        'takeEvery': 13,
        'confName': "fgsfrun17s13",
}]

#full scan
# takeEvery = 1
# confName = "fgsfrun16"

#subsampled 8
# takeEvery = 8
# confName = "fgsfrun16s8"

#subsampled 9
# takeEvery = 9
# confName = "fgsfrun16s9"

#subsampled 10
# takeEvery = 10
# confName = "fgsfrun16s10"

#subsampled 11
# takeEvery = 11
# confName = "fgsfrun16s11"

#Lorentzian fit
possibleFiles = os.listdir(testPath)

eng = matlab.engine.start_matlab()
eng.addpath("~/matlab/toolbox/signal/signal")
eng.addpath("~/matlab/toolbox/stats/stats")
eng.addpath("~/matlab/toolbox/shared/statslib")
eng.cd(r'sim', nargout=0)

printStrings = ["sub meas lf ml"]
for tp in toPrint:
    takeEvery = tp['takeEvery']
    confName = tp['confName']

    allX = []
    allY = []
    maesLF = 0
    i = 0
    found = 0
    while found < testSize:
        if i >= len(possibleFiles):
            maesLF = "na"
            break

        f = h5py.File(os.path.join(testPath, possibleFiles[i]), 'r')
        data = f.get('data_struct')
        f = matlab.double(np.array(data['smp_freqs']).reshape([1,-1])[:,::takeEvery].tolist())
        x = matlab.double(np.array(data['sig'][:, ::takeEvery]).tolist())
        # f = eng.transpose(f)
        # x = eng.transpose(x)
        smooth_data, num_pks, initial_guess, locs = eng.getFitGuess(f,x,0.006, nargout=4)
        if num_pks == 8:
            yprime,params,resnorm,residual,conf = eng.lorentzian_fit_lf(f,x,2,2,8,initial_guess, nargout=5)
            p = np.array([params[0][i] for i in [0, 3, 6, 9, 12, 15, 18, 21]])
            y = np.array(data['peak_locs'])
            maes = abs(p - y).mean()
            # if maes < 100:
            if True:
                maesLF += maes
                # allX.append(np.array(x).reshape(-1))
                allX.append(np.array(data['sig'])[:, ::takeEvery])
                allY.append(y)
                found += 1
            else:
                print(f"Skipped {i}")

        i+=1

    if not maesLF == "na":
        maesLF /= testSize
    else:
        allX = []
        allY = []
        for i in range(testSize):
            f = h5py.File(os.path.join(testPath, possibleFiles[i]), 'r')
            data = f.get('data_struct')
            allX.append(np.array(data['sig'])[:, ::takeEvery])
            allY.append(np.array(data['peak_locs']))

    allX = np.concatenate(allX, axis=0)
    allY = np.concatenate(allY, axis=0)

    # ML
    device = "cuda:0"
    conf = eval('configurations.{}'.format(confName))
    conf.runningPredictions = True
    model, optim, loadEpoch, _ = fun.loadModel(conf, device)
    # dataloaders, _ = fun.processData(conf)
    dataloaders, _ = loaders.custom(conf, allX, allY, batchSize=10, shuffleDataset=False)
    preds = fun.predict(conf, model, dataloaders, loadEpoch, toSave=False, toReturn=True)
    preds = preds['custom']

    maesML = abs(preds['y'] - preds['pred']).mean(axis=1).mean(axis=0)

    printStrings.append(f"{takeEvery} {allX.shape[1]} {maesLF} {maesML}")

for s in printStrings:
    print(s)

eng.quit()
