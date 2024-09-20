#!/usr/bin/env python
# coding: utf-8

#     tuneRunPytorch.py
#     The runner to launch one experiment.
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


#TO RUN ONLY ON GPU 0:   CUDA_VISIBLE_DEVICES=0 python tuneRunPytorch.py ...

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import socket
import configurations
import funPytorch as fun
# import notifier
from datetime import datetime
import time
from pathlib import Path

import ray
from ray import tune,train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

timeFormat = "%Y/%m/%d - %H:%M:%S"

#========================================================
#DEBUG
debug = False
# debug = True
#========================================================

def createRunner(device, baseConf):
    def runner(confDict):
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(os.getcwd())
        # print("-----------------")
        # print(train.get_context().get_trial_dir())
        # print("-----------------")
        # print(train.get_context().get_storage())
        # print("-----------------")
        # storage = train.get_context().get_storage()
        # print(os.path.join(storage.storage_fs_path, storage.experiment_dir_name, storage.trial_dir_name))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        conf = baseConf.copy(confDict)

        if conf.startEpoch == 0:
            model,optim = fun.makeModel(conf, device)
        else:   #maybe this doesn't make sense here
            model,optim,_ = fun.loadModel(conf, device)
        dataloaders, _ = fun.processData(conf)
        fun.runTrain(conf, model, optim, dataloaders, 0, None) #Start allways from epoch 0 (doesn't make sense otherwise in tune)


    return runner


if debug:
    ray.init(local_mode=True, num_cpus=1, num_gpus=0)
else:
    ray.init()

if len(sys.argv) < 2 or len(sys.argv) > 5:
    print("Use {} configName [parProc (def. conf)] [grace (def. conf)] [gpuNum (def. conf)]".format(sys.argv[0]))
else:
    conf = getattr(sys.modules['configurations'], sys.argv[1])
    conf.datasetsDir = os.path.join(os.getcwd(), conf.datasetsDir)
    tuneConf = conf.tuneConf
    
    if len(sys.argv) >= 3:
        parProc = int(sys.argv[2])
    else:
        parProc = conf.tuneParallelProc

    if len(sys.argv) >= 4:
        grace = int(sys.argv[3])
    else:
        grace = conf.tuneGrace

    if len(sys.argv) >= 5:
        device = "cuda:{}".format(sys.argv[3])
    else:
        device = "cuda:{}".format(conf.tuneGpuNum)

    if debug:
        device = "cpu"

    startTime = datetime.now()
    print("====================")
    print("RUN USING {} on device {}".format(sys.argv[1], device))
    print(startTime.strftime(timeFormat))
    print("====================")

    runner = createRunner(device, conf)

    mode = ("max" if conf.bestSign == '>' else "min")

    if conf.tuneHyperOpt:
        scheduler = ASHAScheduler(metric="/".join(["valid",conf.bestKey]), mode=mode, grace_period=grace)#5)
        # searchAlg = HyperOptSearch(tuneConf, metric="/".join(["valid",conf.bestKey]), mode=mode)
        searchAlg = HyperOptSearch(metric="/".join(["valid",conf.bestKey]), mode=mode)
    else:
        scheduler = None
        searchAlg = None

    trialResources = {'cpu': 1., 'gpu': 1./parProc}

    analysis = None
    error = None

    try:
        #================================== To resume
        # analysis = tune.run(runner, config=tuneConf, name=conf.path, scheduler=scheduler, search_alg=searchAlg, resources_per_trial=trialResources, local_dir='tuneOutput', num_samples=conf.tuneNumSamples, resume=True)
        #============================================
        # analysis = tune.run(runner, config=tuneConf, name=conf.path, scheduler=scheduler, search_alg=searchAlg, resources_per_trial=trialResources, local_dir='tuneOutput', num_samples=conf.tuneNumSamples)
        # analysis = tune.run(runner, config=tuneConf, name=conf.path, scheduler=scheduler, search_alg=searchAlg, resources_per_trial=trialResources, local_dir='tuneOutput', num_samples=conf.tuneNumSamples, resume='AUTO')
        tuner = tune.Tuner(  # ③
            tune.with_resources(
                tune.with_parameters(runner),
                resources = trialResources
            ),
            tune_config = tune.TuneConfig(
                scheduler = scheduler,
                search_alg = searchAlg,
                num_samples = conf.tuneNumSamples,
            ),
            run_config = train.RunConfig(
                name = conf.path,
                storage_path = Path('./tuneOutput').resolve()
            ),
            param_space = tuneConf,
        )

        analysis = tuner.fit()

    except Exception as e:
        error = e

    #recover with
    # from ray.tune import Analysis
    # analysis = Analysis("tuneOutput/"+conf.path)


    endTime = datetime.now()

    subject = "Tune Training of {} finished on {}".format(sys.argv[1], socket.gethostname())
    message = "=======\n" \
                "Start: {}\n" \
                "End:   {}\n" \
                "Duration: {}\n" \
                "====================\n".format(
        startTime.strftime(timeFormat),
        endTime.strftime(timeFormat),
        str(endTime-startTime).split('.', 2)[0]
    )

    if not analysis is None:
        bestResult = analysis.get_best_result(scope='all', metric="/".join(["valid",conf.bestKey]), mode=mode)
        message = message + "Best hyperparameters found were: {}\n" \
                "\n" \
                "Best logdir found were: {}\n" \
                "\n" \
                "Best last result found were: {}\n" \
                "====================".format(
            bestResult.config,
            bestResult.path,
            bestResult.metrics["/".join(["valid",conf.bestKey])]
        )

    if not error is None:
        subject = subject + " WITH ERRORS"
        message = message + "\nERROR!!!!\n===========\n{}".format(error)

    print(message)
    # notifier.sendMessage(subject, message)
    with open(os.path.join("tuneOutput",conf.path, "tuneLog.txt"), 'a') as f:
        f.write(message)
        f.write("\n")

    time.sleep(15)
