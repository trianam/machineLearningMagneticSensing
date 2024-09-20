import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matlab.engine


datasets = {
    'data1': {
        'toReference': True,
        'mat7.3': True,
        'measFilename': 'data/galya/1/ESR_meas_100_simu_freqs_3.mat',
        'refFilename': 'data/galya/1/ESR_ref_100_simu_freqs_3.mat',
        'freqFilename': 'data/galya/1/freq.mat',
        'measDataname': 'averaged_meas',
        'refDataname': 'averaged_ref',
        'freqDataname': 'freq',
        'simDatasetCompare': 'datasets/datasetG1.npz',
        'invertPlot': True,
    },
    'data2': {
        'toReference': True,
        'mat7.3': False,
        'measFilename': 'data/galya/2/meas.mat',
        'refFilename': 'data/galya/2/ref.mat',
        'freqFilename': 'data/galya/2/freq.mat',
        'measDataname': 'meas',
        'refDataname': 'ref',
        'freqDataname': 'freq',
        'simDatasetCompare': 'datasets/datasetG2.npz',
        'invertPlot': True,
    },
    'data2b': {
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/2/referenced data.mat',
        'freqFilename': 'data/galya/2/freq.mat',
        'measDataname': 'a',
        'freqDataname': 'freq',
        'simDatasetCompare': 'datasets/datasetG2.npz',
        'invertPlot': True,
    },
    'data3': {
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/3/referenced data.mat',
        'freqFilename': 'data/galya/3/frequency.mat',
        'measDataname': 'a',
        'freqDataname': 'freq',
        'simDatasetCompare': 'datasets/datasetG3.npz',
        'invertPlot': True,
    },
    'data4': {
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/4/referenced data.mat',
        'freqFilename': 'data/galya/4/frequency.mat',
        'measDataname': 'a',
        'freqDataname': 'freq',
        'simDatasetCompare': 'datasets/datasetG4.npz',
        'invertPlot': True,
    },
    'dataSF1': {
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simDatasetCompare': 'datasets/datasetGSF1.npz',
        'predDatasetCompare': 'datasets/datasetPGSF1.npz',
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'dataSF1otf': {
        'title': "Small field 1 full scan",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.037262, 'B_theta':25.691587, 'B_phi':68.56451},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'dataSF1s2': {
        'title': "Small field 1 subsampled every 2",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':37.761448, 'B_theta':25.025959, 'B_phi':67.88369},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 2,
    },
    'dataSF1s3': {
        'title': "Small field 1 subsampled every 3",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':37.97496, 'B_theta':24.918459, 'B_phi':68.04974},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 3,
    },
    'dataSF1s4': {
        'title': "Small field 1 subsampled every 4",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':37.551323, 'B_theta':25.364805, 'B_phi':68.30356},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 4,
    },
    'dataSF1s5': {
        'title': "Small field 1 subsampled every 5",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':0, 'B_theta':0, 'B_phi':0},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 5,
    },
    'dataSF1s6': {
        'title': "Small field 1 subsampled every 6",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':0, 'B_theta':0, 'B_phi':0},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 6,
    },
    'dataSF1s7': {
        'title': "Small field 1 subsampled every 7",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':0, 'B_theta':0, 'B_phi':0},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 7,
    },
    'dataSF1s8': {
        'title': "Small field 1 subsampled every 8",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'simCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':38.17, 'B_theta':25.5755, 'B_phi':67.3352},
        'predCompareOTF': {'N':246, 'center_freq':2877, 'half_window_size':123, 'noiseSigma':0, 'B_mag':37.48815, 'B_theta':25.271967, 'B_phi':68.96297},
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 8,
    },
    #peak prediction
    'dataSF1P': {
        'title': "Small field 1 full scan",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun13/predictions/best-97-real.npz",
        'predComparePeaks': "files/fgsfrun13/predictions/best-97-real.npz", #MSE
        # 'predComparePeaks': "files/fgsfrun13/predictions/last-199-real.npz",
        # 'predComparePeaks': "files/fgsfrun15/predictions/best-122-real.npz", #MAE
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'dataSF1v1P': {
        'title': "Small field 1 full scan",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun14/predictions/best-16-real.npz",
        'predComparePeaks': "files/fgsfrun14/predictions/best-16-real.npz",
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
    },
    'dataSF1Ps8': {
        'title': "Small field 1 subsampled every 8",
        'toReference': False,
        'mat7.3': False,
        'measFilename': 'data/galya/small_field/1/normalized.mat',
        'freqFilename': 'data/galya/small_field/1/freqs.mat',
        'measDataname': 'y',
        'freqDataname': 'x',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun13s8/predictions/best-72-real.npz",
        'predComparePeaks': "files/fgsfrun13s8/predictions/best-72-real.npz",
        'invertPlot': False,
        'cutToFreqs': [2754, 3000],
        'realTakeEvery': 8,
    },
    
    'dataSF1Pd': { #same as dataSF1P but using prepared datasets
        'title': "Small field 1 full scan",
        'measFreqFromDataset': "datasetRGSF1P.npz",
        'measFreqFromDatasetPred': "files/fgsfrun13/predictions/best-97-real.npz",
        'measFreqFromDatasetIndex': 0,
        'measDataname': 'realX',
        'freqDataname': 'realF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun13/predictions/best-97-real.npz",
        'predComparePeaks': "files/fgsfrun13/predictions/best-97-real.npz",
        'invertPlot': False,
    },

    'dataSynP': {
        'title': "Synthetic full scan",
        'measFreqFromDataset': "datasetFRGSF1v2P.npz",
        'measFreqFromDatasetPred': "files/fgsfrun13/predictions/last-199-test.npz",
        'measFreqFromDatasetIndex': 1,
        'measDataname': 'testX',
        'freqDataname': 'testF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun13/predictions/last-199-test.npz",
        'predComparePeaks': "files/fgsfrun13/predictions/last-199-test.npz",
        'invertPlot': False,
    },
    'dataSynPs8': {
        'title': "Synthetic subsampled every 8",
        'measFreqFromDataset': "datasetFRGSF1v2Ps8.npz",
        'measFreqFromDatasetPred': "files/fgsfrun13s8/predictions/last-199-test.npz",
        'measFreqFromDatasetIndex': 1,
        'measDataname': 'testX',
        'freqDataname': 'testF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun13s8/predictions/last-199-test.npz",
        'predComparePeaks': "files/fgsfrun13s8/predictions/last-199-test.npz",
        'invertPlot': False,
    },

    # corrected noise and normalized X
    'dataSynPn': {
        'title': "Synthetic full scan",
        'measFreqFromDataset': "datasetFRGSF1v2nP.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16/predictions/last-199-test.npz",
        # 'measFreqFromDatasetPred': "files/fgsfrun16/predictions/best-213-test.npz",
        'measFreqFromDatasetIndex': 2,
        'measDataname': 'testX',
        'freqDataname': 'testF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16/predictions/last-199-test.npz",
        # 'simComparePeaks': "files/fgsfrun16/predictions/best-213-test.npz",
        'predComparePeaks': "files/fgsfrun16/predictions/last-199-test.npz",
        # 'predComparePeaks': "files/fgsfrun16/predictions/best-213-test.npz",
        'invertPlot': False,
    },
    'dataSynPnT': {
        'title': "Synthetic full scan",
        'measFreqFromDataset': "datasetFRGSF1v2nP.npz",
        # 'measFreqFromDatasetPred': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/last-98-test.npz",
        'measFreqFromDatasetPred': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/best-65-test.npz",
        'measFreqFromDatasetIndex': 2,
        'measDataname': 'testX',
        'freqDataname': 'testF',
        'plotPeaks': True,
        # 'simComparePeaks': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/last-98-test.npz",
        'simComparePeaks': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/best-65-test.npz",
        # 'predComparePeaks': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/last-98-test.npz",
        'predComparePeaks': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/best-65-test.npz",
        'invertPlot': False,
    },
    'dataSynPns8': {
        'title': "Synthetic subsampled every 8",
        'measFreqFromDataset': "datasetFRGSF1v2nPs8.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16s8/predictions/last-199-test.npz",
        'measFreqFromDatasetIndex': 1,
        'measDataname': 'testX',
        'freqDataname': 'testF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16s8/predictions/last-199-test.npz",
        'predComparePeaks': "files/fgsfrun16s8/predictions/last-199-test.npz",
        'invertPlot': False,
    },
    'dataSynPns11': {
        'title': "Synthetic subsampled every 11",
        'measFreqFromDataset': "datasetFRGSF1v2nPs11.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16s11/predictions/last-199-test.npz",
        'measFreqFromDatasetIndex': 1,
        'measDataname': 'testX',
        'freqDataname': 'testF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16s11/predictions/last-199-test.npz",
        'predComparePeaks': "files/fgsfrun16s11/predictions/last-199-test.npz",
        'invertPlot': False,
    },

    'dataSF1Pn': {
        'title': "Small field 1 full scan",
        'measFreqFromDataset': "datasetRGSF1P.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16/predictions/best-213-real.npz",
        'measFreqFromDatasetIndex': 0,
        'measDataname': 'realX',
        'freqDataname': 'realF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16/predictions/best-213-real.npz",
        'predComparePeaks': "files/fgsfrun16/predictions/best-213-real.npz",
        'invertPlot': False,
    },
    'dataSF1PnT': {
        'title': "Small field 1 full scan",
        'measFreqFromDataset': "datasetRGSF1P.npz",
        'measFreqFromDatasetPred': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/best-65-real.npz",
        'measFreqFromDatasetIndex': 0,
        'measDataname': 'realX',
        'freqDataname': 'realF',
        'plotPeaks': True,
        'simComparePeaks': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/best-65-real.npz",
        'predComparePeaks': "tuneOutput/tfgsfrun16/runner_fb9f5d30_106_batchSize=4,dropout=0,hiddenDim=159,hiddenLayers=16,learningRate=0.01,weightDecay=1e-06_2022-11-30_18-46-56/files/predictions/best-65-real.npz",
        'invertPlot': False,
    },
    'dataSF1Pns8': {
        'title': "Small field 1 subsampled every 8",
        'measFreqFromDataset': "datasetRGSF1Ps8.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16s8/predictions/best-29-real.npz",
        'measFreqFromDatasetIndex': 0,
        'measDataname': 'realX',
        'freqDataname': 'realF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16s8/predictions/best-29-real.npz",
        'predComparePeaks': "files/fgsfrun16s8/predictions/best-29-real.npz",
        'invertPlot': False,
    },
    'dataSF1Pns9': {
        'title': "Small field 1 subsampled every 9",
        'measFreqFromDataset': "datasetRGSF1Ps9.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16s9/predictions/best-45-real.npz",
        'measFreqFromDatasetIndex': 0,
        'measDataname': 'realX',
        'freqDataname': 'realF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16s9/predictions/best-45-real.npz",
        'predComparePeaks': "files/fgsfrun16s9/predictions/best-45-real.npz",
        'invertPlot': False,
    },
    'dataSF1Pns10': {
        'title': "Small field 1 subsampled every 10",
        'measFreqFromDataset': "datasetRGSF1Ps10.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16s10/predictions/best-22-real.npz",
        'measFreqFromDatasetIndex': 0,
        'measDataname': 'realX',
        'freqDataname': 'realF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16s10/predictions/best-22-real.npz",
        'predComparePeaks': "files/fgsfrun16s10/predictions/best-22-real.npz",
        'invertPlot': False,
    },
    'dataSF1Pns11': {
        'title': "Small field 1 subsampled every 11",
        'measFreqFromDataset': "datasetRGSF1Ps11.npz",
        'measFreqFromDatasetPred': "files/fgsfrun16s11/predictions/best-44-real.npz",
        'measFreqFromDatasetIndex': 0,
        'measDataname': 'realX',
        'freqDataname': 'realF',
        'plotPeaks': True,
        'simComparePeaks': "files/fgsfrun16s11/predictions/best-44-real.npz",
        'predComparePeaks': "files/fgsfrun16s11/predictions/best-44-real.npz",
        'invertPlot': False,
    },

}

def main(confName):
    data = datasets[confName]

    if 'simComparePeaks' in data and type(data['simComparePeaks']) is str:
        pp = np.load(data['simComparePeaks'])
        if 'measFreqFromDatasetIndex' in data:
            data['simComparePeaks'] = pp['y'][data['measFreqFromDatasetIndex']]
        else:
            data['simComparePeaks'] = pp['y'][0]  # take first and only
    if 'predComparePeaks' in data and type(data['predComparePeaks']) is str:
        pp = np.load(data['predComparePeaks'])
        if 'measFreqFromDatasetIndex' in data:
            data['predComparePeaks'] = pp['pred'][data['measFreqFromDatasetIndex']]
        else:
            data['predComparePeaks'] = pp['pred'][0] #take first and only

    if 'title' in data:
        plt.title(data['title'])
    else:
        plt.title(f"Dataset {confName}")
    plt.xlabel("MHz")

    if 'measFilename' in data and 'freqFilename' in data:
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
            meas = (meas - ref) / np.average(ref)

    elif 'measFreqFromDataset' in data:
        measFreqData = np.load(os.path.join("datasets", data['measFreqFromDataset'])) #freqs from dataset
        measFreqDataPred = np.load(data['measFreqFromDatasetPred']) #signal from pred
        # meas = measFreqData[data['measDataname']][data['measFreqFromDatasetIndex']]
        meas = measFreqDataPred['x'][data['measFreqFromDatasetIndex']]
        freq = measFreqData[data['freqDataname']][data['measFreqFromDatasetIndex']]

    if data['invertPlot']:
        meas = -meas

    if 'cutToFreqs' in data:
        cutIndex = np.argwhere((freq>=data['cutToFreqs'][0]) & (freq<=data['cutToFreqs'][1])).reshape(-1)
        meas = meas[cutIndex]
        freq = freq[cutIndex]

    if 'realTakeEvery' in data:
        freq = freq[::data['realTakeEvery']]
        meas = meas[::data['realTakeEvery']]
    plt.plot(freq, meas, label="Signal")

    if 'simDatasetCompare' in data:
        datasetCompare = np.load(data['simDatasetCompare'])
        simFreq = datasetCompare['trainF'][0]
        simMeas = datasetCompare['trainX'][0]
        if 'simTakeEvery' in data:
            simFreq = simFreq[::data['simTakeEvery']]
            simMeas = simMeas[::data['simTakeEvery']]
        plt.plot(simFreq, simMeas, label=f"Sim B={datasetCompare['trainY'][0]}")
    if 'simCompareOTF' in data:
        eng = matlab.engine.start_matlab()
        simPath = eng.genpath('sim')
        eng.addpath(simPath, nargout=0)
        diamond = eng.mock_diamond2_new_galya(float(data['simCompareOTF']['N']), float(data['simCompareOTF']['center_freq']), float(data['simCompareOTF']['half_window_size']), float(data['simCompareOTF']['noiseSigma']), float(data['simCompareOTF']['B_mag']), float(data['simCompareOTF']['B_theta']), float(data['simCompareOTF']['B_phi']))
        eng.workspace["wDiamond"] = diamond
        simFreq = np.array(eng.eval("wDiamond.smp_freqs")).reshape(-1)
        simMeas = np.array(eng.eval("wDiamond.sig")).reshape(-1)
        if 'simTakeEvery' in data:
            simFreq = simFreq[::data['simTakeEvery']]
            simMeas = simMeas[::data['simTakeEvery']]
        plt.plot(simFreq, simMeas, label=f"Sim B={[float(data['simCompareOTF']['B_mag']), float(data['simCompareOTF']['B_theta']), float(data['simCompareOTF']['B_phi'])]}")

    if 'predDatasetCompare' in data:
        datasetPred = np.load(data['predDatasetCompare'])
        predFreq = datasetPred['trainF'][0]
        predMeas = datasetPred['trainX'][0]
        if 'predTakeEvery' in data:
            predFreq = predFreq[::data['predTakeEvery']]
            predMeas = predMeas[::data['predTakeEvery']]
        plt.plot(predFreq, predMeas, label=f"Pred B={datasetPred['trainY'][0]}")
    if 'predCompareOTF' in data:
        eng = matlab.engine.start_matlab()
        simPath = eng.genpath('sim')
        eng.addpath(simPath, nargout=0)
        diamond = eng.mock_diamond2_new_galya(float(data['predCompareOTF']['N']), float(data['predCompareOTF']['center_freq']), float(data['predCompareOTF']['half_window_size']), float(data['predCompareOTF']['noiseSigma']), float(data['predCompareOTF']['B_mag']), float(data['predCompareOTF']['B_theta']), float(data['predCompareOTF']['B_phi']))
        eng.workspace["wDiamond"] = diamond
        predFreq = np.array(eng.eval("wDiamond.smp_freqs")).reshape(-1)
        predMeas = np.array(eng.eval("wDiamond.sig")).reshape(-1)
        if 'predTakeEvery' in data:
            predFreq = predFreq[::data['predTakeEvery']]
            predMeas = predMeas[::data['predTakeEvery']]
        plt.plot(predFreq, predMeas, label=f"Pred B={[float(data['predCompareOTF']['B_mag']), float(data['predCompareOTF']['B_theta']), float(data['predCompareOTF']['B_phi'])]}")

    if 'plotPeaks' in data and data['plotPeaks']:
        plt.vlines(data['simComparePeaks'], 0, np.max(meas), colors="C0")
        if 'predComparePeaks' in data:
            plt.vlines(data['predComparePeaks'], 0, np.max(meas), colors="C1", linestyles='dashed', label=f"Pred (MAE {sum([abs(t-p) for t,p in zip(data['simComparePeaks'],data['predComparePeaks'])])/len(data['simComparePeaks']):.2f})")

    plt.legend()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    saveTo = f"plots/plot-{confName}.pdf"
    plt.savefig(saveTo, bbox_inches='tight')
    plt.show()
    print(f"Saved to: {saveTo}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)