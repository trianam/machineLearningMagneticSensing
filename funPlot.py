#     funPlot.py
#     Collect the function used to plot the learning curves.
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

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from os import listdir
from os.path import isfile, join, abspath
import math
import numpy as np
import sys
import os
import configurations
import warnings
# from ray.tune import Analysis
from ray.tune import ExperimentAnalysis
import funPytorch
import torch
import loaders

plt.rcParams['figure.dpi'] = 200
plt.rcParams['text.usetex'] = True

def plot(configs, configTitles=None, sets='valid', save=None, colorsFirst=True, title="", limits=None, plotLoss=True, plotMetric=False, plotMetricOverride=None, yLim=None, outerLegend=False, savePlotTable=None, savePlotTableColumns=None):
    keyLoss = 'loss'

    lineStyles = ['solid', 'dashed', 'dotted', 'dashdot']
    lineColors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    if not type(configs) is list:
        configs = [configs]

    if not type(sets) is list:
        sets = [sets]

    if configTitles is None:
        # configTitles = configs
        configTitles = [None] * len(configs)
    elif not type(configTitles) is list:
        configTitles = [configTitles]

    if len(configs)*len(sets) > len(lineStyles)*len(lineColors):
        raise ValueError("Too many curves to plot, {} of max {}.".format(len(configs)*len(sets), len(lineStyles)*len(lineColors)))

    if plotLoss and plotMetric:
        fig = plt.figure(figsize=(8,9))
        #gs = fig.add_gridspec(2,1)
        #axLoss = fig.add_subplot(gs[0, 0])
        #axAcc = fig.add_subplot(gs[1, 0])
        if not outerLegend:
            axLoss = fig.add_axes([0.1, 0.53, 0.85, 0.42])
            axAcc = fig.add_axes([0.1, 0.05, 0.85, 0.42])
        else:
            axLoss = fig.add_axes([0.1, 0.53, 0.6, 0.42])
            axAcc = fig.add_axes([0.1, 0.05, 0.6, 0.42])
    elif plotLoss and not plotMetric:
        fig = plt.figure(figsize=(8,5))
        if not outerLegend:
            axLoss = fig.add_axes([0.1, 0.1, 0.85, 0.8])
        else:
            axLoss = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    elif not plotLoss and plotMetric:
        fig = plt.figure(figsize=(8, 5))
        if not outerLegend:
            axAcc = fig.add_axes([0.1, 0.1, 0.85, 0.8])
        else:
            axAcc = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    else:
        raise Exception("Plot something")

    #fig, (axLoss, axAcc) = plt.subplots(2)
    # metrics = []
    savePlotDict = {}
    i = 0
    iSavePlotTable = 0
    for set in sets:
        for config, configTitle in zip(configs, configTitles):
            myConfig = getattr(sys.modules['configurations'], config)
            # metrics.append(myConfig.plotMetric)

            if myConfig.useTune:
                #use tune to pick best in run
                # analysis = Analysis(join("tuneOutput", myConfig.path))
                analysis = ExperimentAnalysis(abspath(join("tuneOutput", myConfig.path)))
                mode = ("max" if myConfig.bestSign == '>' else "min")
                #print("best hyperparameters for {}: {}".format(config, analysis.get_best_config(metric=myConfig.bestKey, mode=mode)))
                # tunePath = analysis.get_best_logdir(metric=myConfig.bestKey, mode=mode)
                tunePath = analysis.get_best_logdir(metric="/".join(["valid",myConfig.bestKey]), mode=mode)
                expPath = join(tunePath,'files', 'tensorBoard')
            else:
                expPath = join('files', myConfig.path, 'tensorBoard')

            print(f"Load from: {expPath}")

            # keyAcc = "{}_{}".format(metrics[-1], metrics[-1])
            # keyAcc = "{}".format(metrics[-1])
            if plotMetricOverride is None:
                keyAcc = myConfig.plotMetric
            else:
                keyAcc = plotMetricOverride

            if not type(keyAcc) is list:
                keyAcc = [keyAcc]

            # metrics.append(getattr(sys.modules['configurations'], config).plotMetric)
            # keyAcc = "{}_{}".format(metrics[-1], metrics[-1])
            # expPath = join('files', getattr(sys.modules['configurations'], config).path, 'tensorBoard')
            try:
                keys = [f for f in listdir(expPath) if not isfile(join(expPath, f))]
            except FileNotFoundError:
                warnings.warn("Configuration {} not present. Skipping.".format(config))
                continue

            points = {}
            for k in keys:
                eventPathPart = join(expPath, k, set)
                for runPath in sorted([f for f in listdir(eventPathPart) if isfile(join(eventPathPart, f))]):
                    eventPath = join(eventPathPart, runPath)
                    ea = event_accumulator.EventAccumulator(eventPath)
                    ea.Reload()
                    if not k in points:
                        points[k] = [[v.step for v in ea.Scalars(k)], [v.value for v in ea.Scalars(k)]]
                    else:
                        points[k][0].extend([v.step for v in ea.Scalars(k)])
                        points[k][1].extend([v.value for v in ea.Scalars(k)])

            if limits is None:
                valuesLoss = points[keyLoss]
                if plotMetric:
                    valuesAcc = {ka: points[ka] for ka in keyAcc}
            else:
                valuesLoss = [points[keyLoss][i][limits[0]:limits[1]] for i in [0,1]]
                if plotMetric:
                    valuesAcc = {ka: [points[ka][i][limits[0]:limits[1]] for i in [0,1]] for ka in keyAcc}

            if plotLoss:
                if not configTitle is None:
                    label = r"{} - {}".format(configTitle, set)
                else:
                    label = r"{}".format(set)
                linesLoss = axLoss.plot(valuesLoss[0], valuesLoss[1], label=label)
                if not savePlotTable is None:
                    savePlotDict[savePlotTableColumns[iSavePlotTable][0]] = valuesLoss[0]
                    savePlotDict[savePlotTableColumns[iSavePlotTable][1]] = valuesLoss[1]
                    iSavePlotTable += 1

                if colorsFirst:
                    linesLoss[0].set_color(lineColors[i%len(lineColors)])
                    linesLoss[0].set_linestyle(lineStyles[(i//len(lineColors))%len(lineStyles)])
                else:
                    linesLoss[0].set_linestyle(lineStyles[i%len(lineStyles)])
                    linesLoss[0].set_color(lineColors[(i//len(lineStyles))%len(lineColors)])

            if plotMetric:
                for ka in valuesAcc:
                    # print("min {} - {} - {}: {}".format(configTitle, set, ka, min(valuesAcc[ka][1])))
                    if not configTitle is None:
                        label = r"{} - {} - {} (min {:.2})".format(configTitle, set, ka, min(valuesAcc[ka][1]))
                    else:
                        label = r"{} - {} (min {:.2})".format(set, ka, min(valuesAcc[ka][1]))

                    linesAcc = axAcc.plot(valuesAcc[ka][0], valuesAcc[ka][1], label=label)

                    if not savePlotTable is None:
                        savePlotDict[savePlotTableColumns[iSavePlotTable][0]] = valuesAcc[ka][0]
                        savePlotDict[savePlotTableColumns[iSavePlotTable][1]] = valuesAcc[ka][1]
                        iSavePlotTable += 1
                if colorsFirst:
                    linesAcc[0].set_color(lineColors[i%len(lineColors)])
                    linesAcc[0].set_linestyle(lineStyles[(i//len(lineColors))%len(lineStyles)])
                else:
                    linesAcc[0].set_linestyle(lineStyles[i%len(lineStyles)])
                    linesAcc[0].set_color(lineColors[(i//len(lineStyles))%len(lineColors)])

            i += 1


    if plotLoss:
        # axLoss.legend(loc='upper left', bbox_to_anchor=(1, 1),
        #          ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)
        # axLoss.set_title(config)
        axLoss.set_xlabel(r"Epoch")
        axLoss.set_ylabel(r"Loss ({})".format(myConfig.plotLossName))

        # axLoss.set_xticks(np.arange(0, round(axLoss.get_xlim()[1])+10, 10))
        # axLoss.set_xticks(np.arange(round(axLoss.get_xlim()[0]), round(axLoss.get_xlim()[1])+1, 1), minor=True)
        # axLoss.set_yticks(np.arange(0, round(axLoss.get_ylim()[1])+0.05, 0.05))
        # axLoss.set_yticks(np.arange(0, round(axLoss.get_ylim()[1])+0.01, 0.01), minor=True)
        axLoss.grid(which='both')
        axLoss.grid(which='minor', alpha=0.2)
        axLoss.grid(which='major', alpha=0.5)

        if not yLim is None:
            axLoss.set_ylim(yLim)

    if plotMetric:
        # axAcc.legend(loc='upper left', bbox_to_anchor=(1, 1),
        #           ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)
        #axAcc.set_title(title)
        axAcc.set_xlabel(r"Epoch")
        # axAcc.set_ylabel("Metric ({})".format(list(dict.fromkeys(metrics))))
        axAcc.set_ylabel(r"Metric ({})".format(myConfig.plotMetricName))
        #axAcc.set_ylim(0.49,1.)

        #axAcc.set_xticks(np.arange(0, round(axAcc.get_xlim()[1])+10, 10))
        #axAcc.set_xticks(np.arange(round(axAcc.get_xlim()[0]), round(axAcc.get_xlim()[1])+1, 1), minor=True)
        # axAcc.set_yticks(np.arange(0.5, 1.01, 0.1))
        # axAcc.set_yticks(np.arange(0.49, 1.01, 0.01), minor=True)

        axAcc.yaxis.set_minor_locator(AutoMinorLocator())
        axAcc.grid(which='both')
        axAcc.grid(which='minor', alpha=0.2)
        axAcc.grid(which='major', alpha=0.5)

        if not yLim is None:
            axAcc.set_ylim(yLim)

    fig.suptitle(title)

    if plotLoss and plotMetric:
        handles, labels = axLoss.get_legend_handles_labels()
        if not outerLegend:
            fig.legend(handles, labels, bbox_to_anchor=(0.7, 0.9), loc=2, borderaxespad=0.)
        else:
            fig.legend(handles, labels, bbox_to_anchor=(0.72, 0.95), loc=2, borderaxespad=0.)
    elif plotLoss and not plotMetric:
        handles, labels = axLoss.get_legend_handles_labels()
        if not outerLegend:
            fig.legend(handles, labels, bbox_to_anchor=(0.7, 0.85), loc=2, borderaxespad=0.)
        else:
            fig.legend(handles, labels, bbox_to_anchor=(0.72, 0.9), loc=2, borderaxespad=0.)
    elif not plotLoss and plotMetric:
        handles, labels = axAcc.get_legend_handles_labels()
        if not outerLegend:
            fig.legend(handles, labels, bbox_to_anchor=(0.7, 0.85), loc=2, borderaxespad=0.)
        else:
            fig.legend(handles, labels, bbox_to_anchor=(0.72, 0.9), loc=2, borderaxespad=0.)

    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1),
    #              bbox_transform = plt.gcf().transFigure,
    #              ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)

    # plt.legend( handles, labels, loc = 'upper left', bbox_to_anchor = (0.9,-0.1,2,2),
    #         bbox_transform = plt.gcf().transFigure )

    # plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), bbox_transform=plt.gcf().transFigure)

    # fig.subplots_adjust(wspace=2, hspace=2,left=0,top=2,right=2,bottom=0)

    #fig.tight_layout()

    #fig.subplots_adjust(right=2)

    if not savePlotTable is None:
        os.makedirs("img", exist_ok=True)
        plotSaveFileName = f'img/plot-{save}.dat'
        plotKeys = list(savePlotDict.keys())
        plotLen = max([len(savePlotDict[k]) for k in plotKeys])

        with open(plotSaveFileName, 'wt') as f:
            print(" ".join(plotKeys), file=f)
            for i in range(plotLen):
                print(" ".join([str(savePlotDict[k][i]) if i<len(savePlotDict[k]) else 'nan' for k in plotKeys]), file=f)

        print(f"Saved plot data in {plotSaveFileName}")

    fig.show()

    if not save is None:
        os.makedirs("img", exist_ok=True)
        saveFileName = f'img/plot-{save}.pdf'
        fig.savefig(saveFileName, bbox_inches='tight')#, bbox_inches = 'tight')#, pad_inches = 0)
        print(f"Saved plot to: {saveFileName}")

    plt.close()

def plotError(predictions, label=None, save=None, ylim=None):
    if not type(predictions[0]) is list:
        predictions = [predictions]
    if not label is None and not type(label) is list:
        label = [label]
    elif label is None:
        label = [None for _ in predictions]

    plotLegend = False

    for currPredictions, currLabel in zip(predictions, label):
        meas = []
        errs = []
        for predFile in currPredictions:
            pred = np.load(predFile)
            meas.append(len(pred['x'][0])) #take only first as reference
            errs.append(sum([sum([abs(t-p) for t,p in zip(pred['y'][i],pred['pred'][i])])/len(pred['y'][i]) for i in range(len(pred['y']))])/len(pred['y'])) #average of all

        if currLabel is None:
            plt.plot(meas, errs, 'o')
        else:
            plt.plot(meas, errs, 'o', label=currLabel)
            plotLegend = True

    if plotLegend:
        plt.legend()

    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor', linestyle='--')

    plt.xlabel("N. of measurements")
    plt.ylabel("Averaged error (MHz)")

    if not ylim is None:
        plt.ylim(ylim)

    if not save is None:
        os.makedirs("img", exist_ok=True)
        saveFileName = f'img/err-{save}.pdf'
        plt.savefig(saveFileName, bbox_inches='tight')  # , bbox_inches = 'tight')#, pad_inches = 0)
        print(f"Saved plot to: {saveFileName}")

    plt.show()
    plt.close()

def printMetricsOld(configs, printAllConfigs=False):
    sets = ['train', 'valid', 'test']

    if not type(configs) is list:
        configs = [configs]

    bestConfig = None
    points = {}
    for config in configs:
        myConfig = getattr(sys.modules['configurations'], config)
        metric = myConfig.plotMetric
        mode = ("max" if myConfig.bestSign == '>' else "min")

        if myConfig.useTune:
            try:
                #use tune to pick best in run
                # analysis = Analysis(join("tuneOutput", myConfig.path))
                analysis = ExperimentAnalysis(abspath(join("tuneOutput", myConfig.path)))
            except ValueError:
                warnings.warn("Configuration {} not present. Skipping.".format(config))
                continue
            print("best hyperparameters for {}: {}".format(config, analysis.get_best_config(metric=myConfig.bestKey, mode=mode)))
            tunePath = analysis.get_best_logdir(metric=myConfig.bestKey, mode=mode)
            # expPath = join(tunePath,'files','tensorBoard')
            expPath = join(tunePath,'files',config,'tensorBoard')
            print("best logdir: {}".format(expPath))
        else:
            expPath = join('files', config, 'tensorBoard')
        # keyAcc = "{}_{}".format(metric, metric)
        keyAcc = "{}".format(metric)
        try:
            keys = [f for f in listdir(expPath) if not isfile(join(expPath, f))]
        except FileNotFoundError:
            warnings.warn("Configuration {} not present. Skipping.".format(config))
            continue

        points[config] = {}
        for set in sets:
            points[config][set] = {}
            for k in keys:
                eventPathPart = join(expPath, k, set)
                for runPath in sorted([f for f in listdir(eventPathPart) if isfile(join(eventPathPart, f))]):
                    eventPath = join(eventPathPart, runPath)
                    ea = event_accumulator.EventAccumulator(eventPath)
                    ea.Reload()
                    if not k in points[config][set]:
                        points[config][set][k] = [[v.step for v in ea.Scalars(k)], [v.value for v in ea.Scalars(k)]]
                    else:
                        points[config][set][k][0].extend([v.step for v in ea.Scalars(k)])
                        points[config][set][k][1].extend([v.value for v in ea.Scalars(k)])

        bestSign = myConfig.bestSign
        if bestSign == '>':
            bestI = np.argmax(points[config]['valid'][keyAcc][1]) #point where better metric
        else:
            bestI = np.argmin(points[config]['valid'][keyAcc][1]) #point where better metric

        thisConfig = {
            'name': config,
            'epoch': bestI+1, 
            'train': points[config]['train'][keyAcc][1][bestI], 
            'valid': points[config]['valid'][keyAcc][1][bestI], 
            'test': points[config]['test'][keyAcc][1][bestI],
        }
        if printAllConfigs:
            print("{} (epoch {}):\ttrain {:.3};\tvalid {:.3};\ttest {:.3}".format(thisConfig['name'], thisConfig['epoch'], thisConfig['train'], thisConfig['valid'], thisConfig['test']))
            
        if bestConfig is None or (bestSign == '>' and thisConfig['valid']>bestConfig['valid']) or (bestSign == '<' and thisConfig['valid']<bestConfig['valid']):
            bestConfig = thisConfig    

    if not bestConfig is None:
        print("BEST ==== {} (epoch {}):\ttrain {:.3};\tvalid {:.3};\ttest {:.3}".format(bestConfig['name'], bestConfig['epoch'], bestConfig['train'], bestConfig['valid'], bestConfig['test']))


def printMetrics(config):
    myConfig = getattr(sys.modules['configurations'], config)
    argminmax = np.argmax if myConfig.bestSign == '>' else np.argmin
    lessgreat = lambda new, old: ((new>old) if myConfig.bestSign == '>' else (new<old))

    try:
        #use tune to pick best in run
        # analysis = Analysis(join("tuneOutput", myConfig.path))
        analysis = ExperimentAnalysis(abspath(join("tuneOutput", myConfig.path)))
    except ValueError:
        raise ValueError("Configuration {} not present.".format(config))

    trials = analysis.fetch_trial_dataframes()
    configs = analysis.get_all_configs()

    validMetric = (0 if myConfig.bestSign == '>' else np.inf)
    testMetric = np.nan
    trainMetric = np.nan
    epoch = np.nan
    key = ""
    for k in trials:
        trial = trials[k]
        if "/".join(["valid",myConfig.bestKey]) in trial.columns:
            bestIndex = argminmax(trial["/".join(["valid",myConfig.bestKey])])
            if not bestIndex == -1 and lessgreat(trial["/".join(["valid",myConfig.bestKey])][bestIndex], validMetric): # not all nan and the minimum better than previous trials
                validMetric = trial["/".join(["valid",myConfig.bestKey])][bestIndex]
                testMetric = trial["/".join(["test",myConfig.bestKey])][bestIndex]
                trainMetric = trial["/".join(["train",myConfig.bestKey])][bestIndex]
                epoch = bestIndex
                key = k
        else:
            print("{} not present in {}".format("/".join(["valid",myConfig.bestKey]), k))

    print("BEST ==== {}\n\t (epoch {}) {}:\ttrain {:.3};\tvalid {:.3};\ttest {:.3}\n\t{}".format(key, epoch, myConfig.bestKey, trainMetric, validMetric, testMetric, configs[key]))

