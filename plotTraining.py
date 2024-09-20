#     plotTrainingTune.py
#     To plot the learning curves.
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

from funPlot import plot
from itertools import product


# plot(['run1', 'run3', 'trun1', 'run4', 'run2', 'trun2', 'run5'], configTitles=['$\sigma{=}0.002$; 100 freqs', '$\sigma{=}0.001$; 100 freqs', '$\sigma{=}0.001$; 100 freqs; HO', '$\sigma{=}0.0001$; 100 freqs', '$\sigma{=}0.001$; 50 freqs', '$\sigma{=}0.001$; 50 freqs; HO', '$\sigma{=}0.001$; 34 freqs'], sets=['valid'], save=True, plotLoss=False, plotMetric=True)
# plot(['fgrun1'], configTitles=['Galya 1 sim'], sets=['valid'], save='fgrun1', plotLoss=False, plotMetric=True, plotMetricOverride=['mae','mae0','mae1','mae2'])

# plot(['fgsfrun5'], configTitles=['SF 1 full scan'], sets=['train', 'valid'], save='sf1fs', plotLoss=False, plotMetric=True, plotMetricOverride=['mae','mae0','mae1','mae2'])
# plot(['fgsfrun6'], configTitles=['SF 1 subsmp. 2'], sets=['train', 'valid'], save='sf1s2', plotLoss=False, plotMetric=True, plotMetricOverride=['mae','mae0','mae1','mae2'])
# plot(['fgsfrun7'], configTitles=['SF 1 subsmp. 3'], sets=['train', 'valid'], save='sf1s3', plotLoss=False, plotMetric=True, plotMetricOverride=['mae','mae0','mae1','mae2'])
# plot(['fgsfrun8'], configTitles=['SF 1 subsmp. 4'], sets=['train', 'valid'], save='sf1s4', plotLoss=False, plotMetric=True, plotMetricOverride=['mae','mae0','mae1','mae2'])
# plot(['fgsfrun12'], configTitles=['SF 1 subsmp. 8'], sets=['train', 'valid'], save='sf1s8', plotLoss=False, plotMetric=True, plotMetricOverride=['mae','mae0','mae1','mae2'])

#plot(['fgsfrun13'], configTitles=['SF 1 PP full'], sets=['train', 'valid'], save='sf1ppfs', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'])

# plots with augmented data
# plot(['fgsfrun19t'], title='Split 1 - Only real (batchSize=4,dropout=0,hiddenDim=9,hiddenLayers=3,learningRate=0.0100,weightDecay=0)', sets=['train', 'valid'], save='HFs1r', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs1r', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))
# plot(['fgsfrun22t'], title='Split 1 - Augmented (batchSize=4,dropout=0,hiddenDim=39,hiddenLayers=4,learningRate=0.0010,weightDecay=0)', sets=['train', 'valid'], save='HFs1a', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs1a', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))
# plot(['fgsfrun21t'], title='Split 2 - Only real (batchSize=4,dropout=0,hiddenDim=132,hiddenLayers=4,learningRate=0.0100,weightDecay=0)', sets=['train', 'valid'], save='HFs2r', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs2r', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))
# plot(['fgsfrun20t'], title='Split 2 - Augmented (batchSize=2,dropout=0,hiddenDim=551,hiddenLayers=6,learningRate=0.0001,weightDecay=0)', sets=['train', 'valid'], save='HFs2a', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs2a', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))

# plots with interpolated data
# plot(['fgsfrun23ts2'], title='Interpolated subsampling 2', sets=['train', 'valid'], save='HFia2', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20])

# plots with random split
# plot(['fgsfrun24t'], title='Split 3 - Only real (batchSize=4,dropout=0,hiddenDim=58,hiddenLayers=4,learningRate=0.01,weightDecay=0)', sets=['train', 'valid'], save='HFs3r', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs3r', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))
# plot(['fgsfrun25t'], title='Split 3 - Augmented (batchSize=8,dropout=0,hiddenDim=381,hiddenLayers=4,learningRate=0.001,weightDecay=0)', sets=['train', 'valid'], save='HFs3a', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs3a', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))

# plot with 93 synth (to emulate real)
# plot(['fgsfrun28t'], title='Split 2 - 93 synt (batchSize=2,dropout=0,hiddenDim=747,hiddenLayers=5,learningRate=0.01,weightDecay=0.001)', sets=['train', 'valid'], save='HFs2sr', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,40])
# plot with 93 synth (to emulate real), same test valid
# plot(['fgsfrun29t'], title='Split 2 - 93 synt (batchSize=8,dropout=0.2,hiddenDim=222,hiddenLayers=3,learningRate=0.01,weightDecay=0)', sets=['train', 'valid'], save='HFs2sr', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,40])

# plots with 10k augmented
# plot(['fgsfrun26t'], title='Split 1 - Augmented 10k (batchSize=8,dropout=0,hiddenDim=75,hiddenLayers=6,learningRate=0.0001,weightDecay=0)', sets=['train', 'valid'], save='HFs1a10k', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs1a10k', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))
# plot(['fgsfrun27t'], title='Split 2 - Augmented 10k (batchSize=4,dropout=0,hiddenDim=531,hiddenLayers=8,learningRate=0.0001,weightDecay=0)', sets=['train', 'valid'], save='HFs2a10k', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20], savePlotTable='HFs2a10k', savePlotTableColumns=(('trainE','trainL'),('validE','validL')))

# plots with augmented data, different valid test
# plot(['fgsfrun30t'], title='Split 2 diff valid/test - Only real (batchSize=32,dropout=0.2,hiddenDim=909,hiddenLayers=4,learningRate=0.001,weightDecay=0)', sets=['train', 'valid', 'test'], save='HFs2rt', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20])
# plot(['fgsfrun31t'], title='Split 2 diff valid/test - Augmented (batchSize=8,dropout=0,hiddenDim=228,hiddenLayers=5,learningRate=0.001,weightDecay=0)', sets=['train', 'valid', 'test'], save='HFs2at', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,20])

# plots with only synthetic data
plot(['fgsfrun32t'], title='Only synthetic 10k', sets=['train', 'valid', 'test'], save='HFos10k', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,10], savePlotTable='HFos10k', savePlotTableColumns=(('trainE','trainL'),('validE','validL'),('testE','testL')))
plot(['fgsfrun34t'], title='Only synthetic 1k', sets=['train', 'valid', 'test'], save='HFos1k', plotLoss=False, plotMetric=True, plotMetricOverride=['mae'], yLim=[0,10], savePlotTable='HFos1k', savePlotTableColumns=(('trainE','trainL'),('validE','validL'),('testE','testL')))
