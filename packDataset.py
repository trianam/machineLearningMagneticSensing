import os
import numpy as np
import h5py
import argparse
from rich.progress import track

configurations = {
    'dataset1': {
        'trainPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.002/train',
        'validPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.002/valid',
        'testPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.002/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'magnetic_field',
        'takeEvery': 1,
        'saveFilename': 'datasets/dataset1.npz'
    },
    'dataset2': {
        'trainPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/train',
        'validPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/valid',
        'testPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'magnetic_field',
        'takeEvery': 1,
        'saveFilename': 'datasets/dataset2.npz'
    },
    'dataset2f': {
        'trainPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/train',
        'validPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/valid',
        'testPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/dataset2f.npz'
    },
    'dataset3': {
        'trainPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.0001/train',
        'validPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.0001/valid',
        'testPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.0001/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'magnetic_field',
        'takeEvery': 1,
        'saveFilename': 'datasets/dataset3.npz'
    },
    'dataset4': {
        'trainPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/train',
        'validPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/valid',
        'testPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'magnetic_field',
        'takeEvery': 2,
        'saveFilename': 'datasets/dataset4.npz'
    },
    'dataset5': {
        'trainPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/train',
        'validPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/valid',
        'testPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'magnetic_field',
        'takeEvery': 3,
        'saveFilename': 'datasets/dataset5.npz'
    },
    'dataset6': {
        'trainPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/train',
        'validPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/valid',
        'testPath': 'data/large_magnetic_90_120-measures_50-N_100-noise_0.001/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'magnetic_field',
        'takeEvery': 4,
        'saveFilename': 'datasets/dataset6.npz'
    },
    'datasetG1': {
        # 'trainPath': 'data/galya_compatible-measures_50-center_freq_2875-half_window_size_325-N_421-noise_0.0004/train',
        'trainPath': 'data/galya_compatible-measures_50-center_freq_2898-half_window_size_325-N_421-noise_0.0004/train',#   <== align the center with real data
        'validPath': None,
        'testPath': None,
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetG1.npz'
    },
    'datasetG2': {
        # 'trainPath': 'data/galya_compatible-measures_50-center_freq_2875-half_window_size_325-N_651-noise_0.001/train',
        'trainPath': 'data/galya_compatible-measures_50-center_freq_2907-half_window_size_325-N_651-noise_0.001/train',#   <== align the center with real data
        'validPath': None,
        'testPath': None,
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetG2.npz'
    },
    'datasetG3': {
        # 'trainPath': 'data/galya_compatible-measures_50-center_freq_2860-half_window_size_150-N_1000-noise_6e-05/train',
        'trainPath': 'data/galya_compatible-measures_50-center_freq_2880-half_window_size_150-N_1000-noise_6e-05/train',#   <== align the center with real data
        'validPath': None,
        'testPath': None,
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetG3.npz'
    },
    'datasetG4': {
        'trainPath': 'data/galya_compatible-measures_50-center_freq_2870-half_window_size_270-N_191-noise_0.0039/train',
        'validPath': None,
        'testPath': None,
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetG4.npz'
    },
    'datasetFG1': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2875-half_window_size_325-N_421-noise_0.0004/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2875-half_window_size_325-N_421-noise_0.0004/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2875-half_window_size_325-N_421-noise_0.0004/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFG1.npz'
    },
    'datasetGSF1': {
        # 'trainPath': 'data/galya_compatible-measures_50-center_freq_2865-half_window_size_135-N_270-noise_0.0001/train',
        'trainPath': 'data/galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/train',
        'validPath': None,
        'testPath': None,
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetGSF1.npz'
    },
    'datasetFGSF1': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFGSF1.npz'
    },
    'datasetFRGSF1': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/test',
        'validFrom': 'datasets/datasetRGSF1.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFRGSF1.npz'
    },
    'datasetFGSF1v2': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFGSF1v2.npz'
    },
    'datasetFRGSF1v2': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFRGSF1v2.npz'
    },
    'datasetPGSF1': {
        'trainPath': 'data/pred_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/train',
        'validPath': None,
        'testPath': None,
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetPGSF1.npz'
    },
    'datasetFGSF1v2s2': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 2,
        'saveFilename': 'datasets/datasetFGSF1v2s2.npz'
    },
    'datasetFRGSF1v2s2': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1s2.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 2,
        'saveFilename': 'datasets/datasetFRGSF1v2s2.npz'
    },
    'datasetFGSF1v2s3': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 3,
        'saveFilename': 'datasets/datasetFGSF1v2s3.npz'
    },
    'datasetFRGSF1v2s3': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1s3.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 3,
        'saveFilename': 'datasets/datasetFRGSF1v2s3.npz'
    },
    'datasetFGSF1v2s4': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 4,
        'saveFilename': 'datasets/datasetFGSF1v2s4.npz'
    },
    'datasetFRGSF1v2s4': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1s4.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 4,
        'saveFilename': 'datasets/datasetFRGSF1v2s4.npz'
    },
    'datasetFGSF1v2s5': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 5,
        'saveFilename': 'datasets/datasetFGSF1v2s5.npz'
    },
    'datasetFRGSF1v2s5': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1s5.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 5,
        'saveFilename': 'datasets/datasetFRGSF1v2s5.npz'
    },
    'datasetFGSF1v2s6': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 6,
        'saveFilename': 'datasets/datasetFGSF1v2s6.npz'
    },
    'datasetFRGSF1v2s6': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1s6.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 6,
        'saveFilename': 'datasets/datasetFRGSF1v2s6.npz'
    },
    'datasetFGSF1v2s7': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 7,
        'saveFilename': 'datasets/datasetFGSF1v2s7.npz'
    },
    'datasetFRGSF1v2s7': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1s7.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 7,
        'saveFilename': 'datasets/datasetFRGSF1v2s7.npz'
    },
    'datasetFGSF1v2s8': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 8,
        'saveFilename': 'datasets/datasetFGSF1v2s8.npz'
    },
    'datasetFRGSF1v2s8': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1s8.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': ['magnetic_field', 'B_theta', 'B_phi'],
        'takeEvery': 8,
        'saveFilename': 'datasets/datasetFRGSF1v2s8.npz'
    },
    # peak prediction
    'datasetFGSF1v2P': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFGSF1v2P.npz'
    },
    'datasetFRGSF1v2P': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1P.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFRGSF1v2P.npz'
    },
    'datasetFRGSF1P': { # big ranges
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004/test',
        'validFrom': 'datasets/datasetRGSF1P.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFRGSF1P.npz'
    },
    'datasetFRGSF1v2Ps2': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps2.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 2,
        'saveFilename': 'datasets/datasetFRGSF1v2Ps2.npz'
    },
    'datasetFRGSF1v2Ps3': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps3.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 3,
        'saveFilename': 'datasets/datasetFRGSF1v2Ps3.npz'
    },
    'datasetFRGSF1v2Ps4': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps4.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 4,
        'saveFilename': 'datasets/datasetFRGSF1v2Ps4.npz'
    },
    'datasetFRGSF1v2Ps5': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps5.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 5,
        'saveFilename': 'datasets/datasetFRGSF1v2Ps5.npz'
    },
    'datasetFRGSF1v2Ps6': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps6.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 6,
        'saveFilename': 'datasets/datasetFRGSF1v2Ps6.npz'
    },
    'datasetFRGSF1v2Ps7': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps7.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 7,
        'saveFilename': 'datasets/datasetFRGSF1v2Ps7.npz'
    },
    'datasetFRGSF1v2Ps8': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.0004-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps8.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 8,
        'saveFilename': 'datasets/datasetFRGSF1v2Ps8.npz'
    },
    # with realistic noise
    'datasetFRGSF1v2nP': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': 'datasets/datasetRGSF1P.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFRGSF1v2nP.npz'
    },
    'datasetFRGSF1v2nPs8': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps8.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 8,
        'saveFilename': 'datasets/datasetFRGSF1v2nPs8.npz'
    },
    'datasetFRGSF1v2nPs9': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps9.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 9,
        'saveFilename': 'datasets/datasetFRGSF1v2nPs9.npz'
    },
    'datasetFRGSF1v2nPs10': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps10.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 10,
        'saveFilename': 'datasets/datasetFRGSF1v2nPs10.npz'
    },
    'datasetFRGSF1v2nPs11': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': None,
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': 'datasets/datasetRGSF1Ps11.npz',
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 11,
        'saveFilename': 'datasets/datasetFRGSF1v2nPs11.npz'
    },



    # synth validation
    'datasetFGSF1v2nP': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 1,
        'saveFilename': 'datasets/datasetFGSF1v2nP.npz'
    },
    'datasetFGSF1v2nPs2': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 2,
        'saveFilename': 'datasets/datasetFGSF1v2nPs2.npz'
    },
    'datasetFGSF1v2nPs3': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 3,
        'saveFilename': 'datasets/datasetFGSF1v2nPs3.npz'
    },
    'datasetFGSF1v2nPs4': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 4,
        'saveFilename': 'datasets/datasetFGSF1v2nPs4.npz'
    },
    'datasetFGSF1v2nPs5': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 5,
        'saveFilename': 'datasets/datasetFGSF1v2nPs5.npz'
    },
    'datasetFGSF1v2nPs6': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 6,
        'saveFilename': 'datasets/datasetFGSF1v2nPs6.npz'
    },
    'datasetFGSF1v2nPs7': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 7,
        'saveFilename': 'datasets/datasetFGSF1v2nPs7.npz'
    },
    'datasetFGSF1v2nPs8': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 8,
        'saveFilename': 'datasets/datasetFGSF1v2nPs8.npz'
    },
    'datasetFGSF1v2nPs9': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 9,
        'saveFilename': 'datasets/datasetFGSF1v2nPs9.npz'
    },
    'datasetFGSF1v2nPs10': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 10,
        'saveFilename': 'datasets/datasetFGSF1v2nPs10.npz'
    },
    'datasetFGSF1v2nPs11': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 11,
        'saveFilename': 'datasets/datasetFGSF1v2nPs11.npz'
    },
    'datasetFGSF1v2nPs12': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 12,
        'saveFilename': 'datasets/datasetFGSF1v2nPs12.npz'
    },
    'datasetFGSF1v2nPs13': {
        'trainPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/train',
        'validPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/valid',
        'testPath': 'data/full_galya_compatible-measures_50-center_freq_2877-half_window_size_123-N_246-noise_0.00043304-v2/test',
        'validFrom': None,
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 13,
        'saveFilename': 'datasets/datasetFGSF1v2nPs13.npz'
    },
}

def main(confName):
    myConf = configurations[confName]

    if not type(myConf['yDataName']) is list and not myConf['yDataName']=='peak_locs':
        myConf['yDataName'] = [myConf['yDataName']]

    trainF = []
    trainX = []
    trainY = []
    validF = []
    validX = []
    validY = []
    testF = []
    testX = []
    testY = []

    for dataSplit, dataPath, listF, listX, listY in [('train', myConf['trainPath'], trainF, trainX, trainY), ('valid', myConf['validPath'], validF, validX, validY), ('test', myConf['testPath'], testF, testX, testY)]:
        # for filename in os.listdir(dataPath):
        if not dataPath is None:
            for filename in track(os.listdir(dataPath), f"Loading {dataSplit} files"):
                f = h5py.File(os.path.join(dataPath, filename), 'r')
                data = f.get('data_struct')
                listF.append(np.array(data[myConf['fDataName']]).reshape([1,-1])[:,::myConf['takeEvery']])
                listX.append(np.array(data[myConf['xDataName']])[:,::myConf['takeEvery']])
                if myConf['yDataName'] == 'peak_locs':
                    listY.append(np.array(data[myConf['yDataName']]))
                else:
                    listY.append(np.array([[data[y][0][0] for y in myConf['yDataName']]]))
        else:
            listF.append(np.array([]))
            listX.append(np.array([]))
            listY.append(np.array([]))

    trainF = np.concatenate(trainF, axis=0)
    trainX = np.concatenate(trainX, axis=0)
    trainY = np.concatenate(trainY, axis=0)
    validF = np.concatenate(validF, axis=0)
    validX = np.concatenate(validX, axis=0)
    validY = np.concatenate(validY, axis=0)
    testF = np.concatenate(testF, axis=0)
    testX = np.concatenate(testX, axis=0)
    testY = np.concatenate(testY, axis=0)

    if not myConf['validFrom'] is None:
        otherDataset = np.load(myConf['validFrom'])
        validF = otherDataset['realF']
        validX = otherDataset['realX']
        validY = otherDataset['realY']

    np.savez_compressed(myConf['saveFilename'], trainF=trainF, trainX=trainX, trainY=trainY, validF=validF, validX=validX, validY=validY, testF=testF, testX=testX, testY=testY)
    print(f"Saved in {myConf['saveFilename']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
