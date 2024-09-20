#     loaders.py
#     The class and function to create the Pytorch dataset and dataloader.
#     Copyright (C) 2021  Stefano Martina (stefano.martina@unifi.it)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import h5py
import os

class DatasetNpz(torch.utils.data.Dataset):
    def __init__(self, conf, mySet=None, getFreqs=False):
        self.conf = conf
        self.getFreqs = getFreqs

        # if 'datasets' in os.listdir('.'):
        #     basePath = "datasets"
        # else:
        #     basePath = "../../../datasets" #go up for tune
        #
        # fileDataset = np.load(os.path.join(basePath,conf.dataset))
        fileDataset = np.load(os.path.join(conf.datasetsDir,conf.dataset))

        self.data = {}

        x = fileDataset[mySet+'X'].astype(np.float32)
        y = fileDataset[mySet+'Y'].astype(np.float32)

        if x.shape[0] < conf.batchSize: #when use small real datasets
            x = np.repeat(x, conf.batchSize, axis=0)
            y = np.repeat(y, conf.batchSize, axis=0)

        if conf.normalizeX:
            for i in range(x.shape[0]):
                x[i] -= np.min(x[i])
                x[i] /= np.max(x[i])

        if conf.normalizeY:
            for i in range(y.shape[1]):
                y[:,i] -= conf.minMax[i][0]
                y[:,i] /= conf.minMax[i][1] - conf.minMax[i][0]

        self.data['x'] = x
        self.data['y'] = y

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        toReturn = {
            'x': self.data['x'][idx],
            'y': self.data['y'][idx],
        }

        return toReturn

def npz(conf):
    datasets = {s: DatasetNpz(conf, s) for s in conf.splits}
    loaders = {s: torch.utils.data.DataLoader(datasets[s], batch_size=conf.batchSize, shuffle=conf.shuffleDataset) for s in conf.splits}
    
    return loaders, datasets


class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self, conf, x, y, batchSize):
        self.conf = conf
        self.data = {}

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        if x.shape[0] < batchSize:  # when use small real datasets
            x = np.repeat(x, batchSize, axis=0)
            y = np.repeat(y, batchSize, axis=0)

        if conf.normalizeX:
            for i in range(x.shape[0]):
                x[i] -= np.min(x[i])
                x[i] /= np.max(x[i])

        if conf.normalizeY:
            for i in range(y.shape[1]):
                y[:, i] -= conf.minMax[i][0]
                y[:, i] /= conf.minMax[i][1] - conf.minMax[i][0]

        self.data['x'] = x
        self.data['y'] = y

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        toReturn = {
            'x': self.data['x'][idx],
            'y': self.data['y'][idx],
        }

        return toReturn


def custom(conf, x, y, batchSize=None, shuffleDataset=None):
    batchSize = conf.batchSize if batchSize is None else batchSize
    shuffleDataset = conf.shuffleDataset if shuffleDataset is None else shuffleDataset

    datasets = {'custom': DatasetCustom(conf, x, y, batchSize)}
    loaders = {'custom': torch.utils.data.DataLoader(datasets['custom'],
                                                     batch_size=batchSize,
                                                     shuffle=shuffleDataset,
                                                     num_workers=4)}

    return loaders, datasets

