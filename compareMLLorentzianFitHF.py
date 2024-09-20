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

thisConfigurations = {
    'fgsfrun18':{
        'maxValue': None,
        'testSize': 80,
        'testData': "datasets/datasetFRGHF1P.npz",
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.005,
        'minPeakProminence': 0,
        'smoothSpan': 0.06,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [
                    {
                        'takeEvery': 1,
                        'confName': "fgsfrun18",
                    }
                  ]+[
                    {
                        'takeEvery': s,
                        'confName': f"fgsfrun18s{s}",
                    } for s in list(range(2,21))+[40]],
        'saveFilename': "dats/fgsfrun18.dat",
    },
    'fgsfrun18S':{
        'maxValue': None,
        'testSize': 80,
        'testData': "datasets/datasetFRGHF1P.npz",
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.005,
        'minPeakProminence': 0,
        'smoothSpan': 0.06,
        'testSet': 'test',
        'freqName': 'freq',
        'toPrint': [
                    {
                        'takeEvery': 1,
                        'confName': "fgsfrun18",
                    }
                  ]+[
                    {
                        'takeEvery': s,
                        'confName': f"fgsfrun18s{s}",
                    } for s in list(range(2,21))+[40]],
        'saveFilename': 'dats/fgsfrun18S.dat',
    },

    'fgsfrun19':{
        'maxValue': None,
        'testSize': None,
        'testData': "datasets/datasetAHF1P.npz",
        'skipLF': True,
        'skipML': False,
        'minPeakHeight': 0.005,
        'minPeakProminence': 0,
        'smoothSpan': 0.06,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [
                    {
                        'takeEvery': 1,
                        'confName': "fgsfrun19",
                    }
                    ],
                  # ]+[
                  #   {
                  #       'takeEvery': s,
                  #       'confName': f"fgsfrun19s{s}",
                  #   } for s in list(range(2,21))+[40]],
        'saveFilename': "dats/fgsfrun19.dat",
    },
    'fgsfrun20tOnlyML':{
        'maxValue': None,
        'testSize': None,
        'testData': "datasets/datasetA4HF1P.npz",
        'skipLF': True,
        'skipML': False,
        'minPeakHeight': 0.005,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.06,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [
                    {
                        'takeEvery': 1,
                        'confName': "fgsfrun20t",
                    }
                  ]+[
                    {
                        'takeEvery': s,
                        'confName': f"fgsfrun20ts{s}",
                    } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun20tOnlyML.dat",
    },
    'fgsfrun20t':{
        'maxValue': None,
        # 'testSize': None,
        'testSize': 10,
        'testData': "datasets/datasetA4HF1P.npz",
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.005,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.06,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [
                    {
                        'takeEvery': 1,
                        'confName': "fgsfrun20t",
                    }
                  ]+[
                    {
                        'takeEvery': s,
                        'confName': f"fgsfrun20ts{s}",
                    } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun20t.dat",
    },

    'fgsfrun20tf':{
        'maxValue': 100,
        'testSize': None,
        'testData': "datasets/datasetA4HF1P.npz",
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [
                    {
                        'takeEvery': 1,
                        'confName': "fgsfrun20t",
                    }
                  ]+[
                    {
                        'takeEvery': s,
                        'confName': f"fgsfrun20ts{s}",
                    } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun20tf.dat",
    },

    'fgsfrun23t':{
        'maxValue': None,
        # 'testSize': None,
        'testSize': 10,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.005,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.06,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23ts{s}",
                        'overwriteSub': s,
                    # } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
                    # } for s in [2,3,4,5,6,7,8,9,10,11,12,13,14]]+[{
                    } for s in [2,3,4,5,6,7,8,9]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23s{s}",
                        'overwriteSub': s,
                    } for s in list(range(10,12))] + [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23ts{s}",
                        'overwriteSub': s,
                    } for s in [12]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23s{s}",
                        'overwriteSub': s,
                    } for s in list(range(13,26))+[27,28,29,30,32,34,36,38,40]],
        'saveFilename': "dats/fgsfrun23t.dat",
    },

    'fgsfrun23tf':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23ts{s}",
                        'overwriteSub': s,
                    # } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
                    # } for s in [2,3,4,5,6,7,8,9,10,11,12,13,14]]+[{
                    } for s in [2,3,4,5,6,7,8,9]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23s{s}",
                        'overwriteSub': s,
                    } for s in list(range(10,12))] + [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23ts{s}",
                        'overwriteSub': s,
                    } for s in [12]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23s{s}",
                        'overwriteSub': s,
                    } for s in list(range(13,26))+[27,28,29,30,32,34,36,38,40]],
        'saveFilename': "dats/fgsfrun23tf.dat",
    },
    
    'fgsfrun23tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun20t",
                        'overwriteSub': s,
                    # } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
                    # } for s in [2,3,4,5,6,7,8,9,10,11,12,13,14]]+[{
                    } for s in [2,3,4,5,6,7,8,9]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun20t",
                        'overwriteSub': s,
                    } for s in list(range(10,12))] + [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun20t",
                        'overwriteSub': s,
                    } for s in [12]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun20t",
                        'overwriteSub': s,
                    } for s in list(range(13,26))+[27,28,29,30,32,34,36,38,40]],
        'saveFilename': "dats/fgsfrun23tf1.dat",
    },

    'fgsfrun23tfo':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.005,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.06,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23ts{s}",
                        'overwriteSub': s,
                    # } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
                    # } for s in [2,3,4,5,6,7,8,9,10,11,12,13,14]]+[{
                    } for s in [2,3,4,5,6,7,8,9]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23s{s}",
                        'overwriteSub': s,
                    } for s in list(range(10,12))] + [{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23ts{s}",
                        'overwriteSub': s,
                    } for s in [12]]+[{
                        'testData': f"datasets/datasetIA4HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun23s{s}",
                        'overwriteSub': s,
                    } for s in list(range(13,26))+[27,28,29,30,32,34,36,38,40]],
        'saveFilename': "dats/fgsfrun23tfo.dat",
    },

    # ========== only synth
    'fgsfrun32tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1P.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32tPred",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32tPred",
                        'overwriteSub': s,
                    } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32tf1.dat",
    },

    # ========== only synth, distance corrected
    'fgsfrun32ctf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pc.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pcs{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': s,
                    } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32ctf1.dat",
    },

    # ========== only synth, distance corrected more
    'fgsfrun32cctf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pcc.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32cct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pccs{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32cct",
                        'overwriteSub': s,
                    } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32cctf1.dat",
    },

    # ========== only synth, defined width
    'fgsfrun32w15tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw15.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw15s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w15tf1.dat",
    },

    'fgsfrun32w10tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw10.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw10s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10tf1.dat",
    },

    'fgsfrun32w6tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw6.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w6tf1.dat",
    },

    # ========== only synth, defined SNR
    'fgsfrun32w10n10tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn10.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn10s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10n10tf1.dat",
    },

    'fgsfrun32w10n6tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn6.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10n6tf1.dat",
    },

    'fgsfrun32w10n2tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn2.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn2s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ct",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10n2tf1.dat",
    },

    # ========== only synth, defined SNR, trained on fixed SNR
    'fgsfrun32w10n10tf1F':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn10.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctN6",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn10s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctN6",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10n10tf1F.dat",
    },

    'fgsfrun32w10n6tf1F':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn6.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctN6",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctN6",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10n6tf1F.dat",
    },

    'fgsfrun32w10n2tf1F':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn2.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctN6",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn2s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctN6",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10n2tf1F.dat",
    },

# ========== only synth, defined width, retrained
    'fgsfrun32w15tf1t':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw15.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctw15",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw15s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctw15",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w15tf1t.dat",
    },

    'fgsfrun32w6tf1t':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw6.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctw6",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctw6",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w6tf1t.dat",
    },

# ========== only synth, defined width, retrained on range
    'fgsfrun32w15tf1tr':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw15.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw15s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w15tf1tr.dat",
    },

    'fgsfrun32w10tf1tr': {
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,  # .001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
            'testData': f"datasets/datasetIA6HF1Pw10.npz",
            'takeEvery': 1,
            'confName': "fgsfrun32ctwR",
            'overwriteSub': 1,
        }] + [{
            'testData': f"datasets/datasetIA6HF1Pw10s{s}.npz",
            'takeEvery': 1,
            'confName': "fgsfrun32ctwR",
            'overwriteSub': s,
        } for s in list(range(2, 11))],  # to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10tf1tr.dat",
    },

    'fgsfrun32w6tf1tr':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw6.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w6tf1tr.dat",
    },

# ========== only synth, defined width, retrained on wider range
    'fgsfrun32w15tf1tr2':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw15.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw15s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w15tf1tr2.dat",
    },

    'fgsfrun32w10tf1tr2': {
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,  # .001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
            'testData': f"datasets/datasetIA6HF1Pw10.npz",
            'takeEvery': 1,
            'confName': "fgsfrun32ctwR2",
            'overwriteSub': 1,
        }] + [{
            'testData': f"datasets/datasetIA6HF1Pw10s{s}.npz",
            'takeEvery': 1,
            'confName': "fgsfrun32ctwR2",
            'overwriteSub': s,
        } for s in list(range(2, 11))],  # to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10tf1tr2.dat",
    },

    'fgsfrun32w6tf1tr2':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pw6.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pw6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w6tf1tr2.dat",
    },

# ========== only synth, defined width, retrained on wider range, same parameters, only where LF is working
    'fgsfrun32w15tf1tr2sp':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmw15.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmw15s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w15tf1tr2spn.dat",
    },
    'fgsfrun32w10tf1tr2sp':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmw10.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmw10s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10tf1tr2spn.dat",
    },
    'fgsfrun32w6tf1tr2sp':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmw6.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmw6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w6tf1tr2spn.dat",
    },

# ========== only synth, defined width, retrained on wider range, same parameters, only where LF is working, fixed noise
    'fgsfrun32w15tf1tr2spfn':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmw15fn.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmw15fns{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w15tf1tr2spfn.dat",
    },
    'fgsfrun32w10tf1tr2spfn':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmw10fn.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmw10fns{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w10tf1tr2spfn.dat",
    },
    'fgsfrun32w6tf1tr2spfn':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmw6fn.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmw6fns{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32ctwR2",
                        'overwriteSub': s,
                    } for s in list(range(2,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun32w6tf1tr2spfn.dat",
    },


# ========== only synth, defined width, retrained on wider range, no interpolation and only LF
    'fgsfrun32w15tf1tr2ni':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        # 'testData': f"datasets/datasetIA6HF1Pw15.npz",
                        'testData': f"datasets/datasetFGHF1Pw15.npz",
                        'takeEvery': s,
                        'confName': "FAKE",
                    } for s in list(range(1,11))],
        'saveFilename': "dats/fgsfrun32w15tf1tr2ni.dat",
    },

    'fgsfrun32w10tf1tr2ni':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        # 'testData': f"datasets/datasetIA6HF1Pw10.npz",
                        'testData': f"datasets/datasetFGHF1Pw10.npz",
                        'takeEvery': s,
                        'confName': "FAKE",
                    } for s in list(range(1,11))],
        'saveFilename': "dats/fgsfrun32w10tf1tr2ni.dat",
    },

    'fgsfrun32w6tf1tr2ni':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        # 'testData': f"datasets/datasetIA6HF1Pw6.npz",
                        'testData': f"datasets/datasetFGHF1Pw6.npz",
                        'takeEvery': s,
                        'confName': "FAKE",
                    } for s in list(range(1,11))],
        'saveFilename': "dats/fgsfrun32w6tf1tr2ni.dat",
    },


    #subsampling
    'fgsfrun32tf1s':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1P.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun32tPred",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetFGHF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': f"fgsfrun32ts{s}",
                        'overwriteSub': 1,
                    } for s in [3,5,8,10,15,20,30]],
        'saveFilename': "dats/fgsfrun32tf1s.dat",
    },

    'fgsfrun34tf1':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': False,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA5HF1P.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun34tPred",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA5HF1Ps{s}.npz",
                        'takeEvery': 1,
                        'confName': "fgsfrun34tPred",
                        'overwriteSub': s,
                    } for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]], #to avoid doubles with same number of measurements
        'saveFilename': "dats/fgsfrun34tf1.dat",
    },

    #=========================== fixed noise level, only LF
    'noiseErr10':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn10.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn10s{s}.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': s,
                    } for s in list(range(2,11))],
        'saveFilename': "dats/noiseErr10.dat",
    },

    'noiseErr6':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn6.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': s,
                    } for s in list(range(2,11))],
        'saveFilename': "dats/noiseErr6.dat",
    },

    'noiseErr2':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pn2.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pn2s{s}.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': s,
                    } for s in list(range(2,11))],
        'saveFilename': "dats/noiseErr2.dat",
    },

    #=========================== fixed noise level, only LF not interpolated
    'noiseErr10ni':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetFGHF1Pn10.npz",
                        'takeEvery': s,
                        'confName': "FAKE",
                    } for s in list(range(1,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/noiseErr10ni.dat",
    },

    'noiseErr6ni':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetFGHF1Pn6.npz",
                        'takeEvery': s,
                        'confName': "FAKE",
                    } for s in list(range(1,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/noiseErr6ni.dat",
    },

    'noiseErr2ni':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetFGHF1Pn2.npz",
                        'takeEvery': s,
                        'confName': "FAKE",
                    } for s in list(range(1,11))], #to avoid doubles with same number of measurements
        'saveFilename': "dats/noiseErr2ni.dat",
    },

    #=========================== fixed noise level, only LF, same parameters, only where LF is working
    'noiseErr10sp':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmn10.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmn10s{s}.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': s,
                    } for s in list(range(2,11))],
        'saveFilename': "dats/noiseErr10spn.dat",
    },
    'noiseErr6sp':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmn6.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmn6s{s}.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': s,
                    } for s in list(range(2,11))],
        'saveFilename': "dats/noiseErr6spn.dat",
    },
    'noiseErr2sp':{
        'maxValue': 100,
        'testSize': None,
        'skipLF': False,
        'skipML': True,
        'minPeakHeight': 0.004,
        'minPeakProminence': 0,#.001,
        'smoothSpan': 0.025,
        'testSet': 'valid',
        'freqName': 'freq',
        'toPrint': [{
                        'testData': f"datasets/datasetIA6HF1Pmn2.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': 1,
                    }] + [{
                        'testData': f"datasets/datasetIA6HF1Pmn2s{s}.npz",
                        'takeEvery': 1,
                        'confName': "FAKE",
                        'overwriteSub': s,
                    } for s in list(range(2,11))],
        'saveFilename': "dats/noiseErr2spn.dat",
    },
}

thisConfigurations['fgsfrun23tOnlyML'] = {
    **thisConfigurations['fgsfrun23t'],
    'testSize': None,
    'skipLF': True,
    'saveFilename': "dats/fgsfrun23tOnlyML.dat",
}

def main(confName):
    conf = thisConfigurations[confName]

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

    printStrings = ["sub meas lfworks lf lfstd lfconf lfconfstd lfnorm lfnormstd lfconfhalf lfconfhalfstd lfnormhalf lfnormhalfstd ml mlstd"]
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

        allX = []
        allY = []
        maesLF = 0
        maesLFsd = 0
        confLF = 0
        confLFsd = 0
        normLF = 0
        normLFsd = 0
        confHalfLF = 0
        confHalfLFsd = 0
        normHalfLF = 0
        normHalfLFsd = 0
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
                confidenceHalf = confidence / 2
                p = np.array([params[0][i] for i in [0, 3, 6, 9, 12, 15, 18, 21]])
                y = data[f"{conf['testSet']}Y"][iSample]
                maes = abs(p - y).mean()
                # if maes < 100:
                if True:
                    maesLF += maes
                    maesLFsd += maes * maes

                    confidence = np.sqrt((p-y)**2 + confidence**2).mean()
                    confLF += confidence
                    confLFsd += confidence * confidence

                    confidenceHalf = np.sqrt((p - y) ** 2 + confidenceHalf ** 2).mean()
                    confHalfLF += confidenceHalf
                    confHalfLFsd += confidenceHalf * confidenceHalf

                    # allX.append(np.array(x).reshape(-1))
                    allX.append(data[f"{conf['testSet']}X"][iSample,::takeEvery])
                    allY.append(y)
                    found += 1
                else:
                    print(f"Skipped {iSample}")
            # else:
            #     print(f"Skipped {iSample}")

            iSample+=1

        if found == 0:
            maesLF = 'nan'
            maesLFsd = 'nan'
            confLF = 'nan'
            confLFsd = 'nan'
            normLF = 'nan'
            normLFsd = 'nan'
            confHalfLF = 'nan'
            confHalfLFsd = 'nan'
            normHalfLF = 'nan'
            normHalfLFsd = 'nan'
        else:
            maesLF /= found
            maesLFsd = math.sqrt((maesLFsd / found) - (maesLF * maesLF))

            confLF /= found
            confLFsd = math.sqrt((confLFsd / found) - (confLF * confLF))
            normLF = confLF / math.sqrt(found/testSize)
            normLFsd = confLFsd / math.sqrt(found/testSize)

            confHalfLF /= found
            confHalfLFsd = math.sqrt((confHalfLFsd / found) - (confHalfLF * confHalfLF))
            normHalfLF = confHalfLF / math.sqrt(found/testSize)
            normHalfLFsd = confHalfLFsd / math.sqrt(found/testSize)

        if foundLess:
            allX = []
            allY = []
            for i in range(testSize):
                allX.append(data[f"{conf['testSet']}X"][i,::takeEvery])
                allY.append(data[f"{conf['testSet']}Y"][i])

        allX = np.stack(allX)
        allY = np.stack(allY)

        # ML
        if not conf['skipML']:
            print(f"Set {iTp+1}/{len(conf['toPrint'])} - Calculating ML")

            device = "cuda:0"
            confML = eval('configurations.{}'.format(confName))
            confML.runningPredictions = True
            model, optim, loadEpoch, _ = fun.loadModel(confML, device)
            # dataloaders, _ = fun.processData(conf)
            dataloaders, _ = loaders.custom(confML, allX, allY, batchSize=10, shuffleDataset=False)
            preds = fun.predict(confML, model, dataloaders, loadEpoch, toSave=False, toReturn=True)
            preds = preds['custom']

            maesML = abs(preds['y'] - preds['pred']).mean(axis=1).mean(axis=0)
            maesMLsd = abs(preds['y'] - preds['pred']).mean(axis=1).std(axis=0)
        else:
            maesML = "nan"
            maesMLsd = "nan"

        if not conf['maxValue'] is None:
            if maesLF != 'nan' and maesLF > conf['maxValue']:
                maesLF = conf['maxValue']
                maesLFsd = 0 # assume that the lower sd point is out of scale anyway
            if maesML != 'nan' and maesML > conf['maxValue']:
                maesML = conf['maxValue']
                maesMLsd = 0 # assume that the lower sd point is out of scale anyway

        if 'overwriteSub' in tp:
            printStrings.append(f"{tp['overwriteSub']} {math.ceil(allX.shape[1] / tp['overwriteSub'])} {found} {maesLF} {maesLFsd} {confLF} {confLFsd} {normLF} {normLFsd} {confHalfLF} {confHalfLFsd} {normHalfLF} {normHalfLFsd} {maesML} {maesMLsd}")
        else:
            printStrings.append(f"{takeEvery} {allX.shape[1]} {found} {maesLF} {maesLFsd} {confLF} {confLFsd} {normLF} {normLFsd} {confHalfLF} {confHalfLFsd} {normHalfLF} {normHalfLFsd} {maesML} {maesMLsd}")

    print(f"Saving to {conf['saveFilename']}")
    print("----------------")
    Path(os.path.dirname(conf['saveFilename'])).mkdir(parents=True, exist_ok=True)
    with open(conf['saveFilename'], 'wt') as f:
        for s in printStrings:
            print(s)
            print(s, file=f)

    eng.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Configuration name')

    args = parser.parse_args()
    main(args.configName)