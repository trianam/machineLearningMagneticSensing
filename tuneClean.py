#!/usr/bin/env python

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import shutil
import socket
import configurations
import funPytorch as fun
# import notifier
from datetime import datetime

if __name__ == '__main__':
    timeFormat = "%Y/%m/%d - %H:%M:%S"

    if len(sys.argv) != 2:
        print("Use {} configName".format(sys.argv[0]))
    else:
        # conf = getattr(sys.modules['configurations'], sys.argv[1])
        conf = eval('configurations.{}'.format(sys.argv[1]))

        if not conf.has("useTune") or not conf.useTune:
            raise ValueError("Configuration without Tune")

        conf.runningPredictions = True
        # conf = conf.copy({"runningPredictions": True})

        startTime = datetime.now()

        print("====================")
        print("CLEAN {}".format(sys.argv[1]))
        print(startTime.strftime(timeFormat))
        print("====================")
        print("======= LOAD PATHS")
        filesPathLog = fun.tuneBestRunnerPath(conf, getFromLog=True)
        filesPath = fun.tuneBestRunnerPath(conf, getFromLog=False)

        if filesPathLog is None:
            # cont = input("Tune log file not found, continue without it? [y/n]:")
            # if cont != "y":
            #     raise Exception("Tune log file not found")
            print("Tune log file not found")
            pathsToSave = [filesPath]
        else:
            pathsToSave = [filesPathLog, filesPath]

        allPaths = list(map(lambda d: os.path.join("tuneOutput", conf.path, d), [d for d in os.listdir(os.path.join("tuneOutput", conf.path)) if d.startswith("runner")]))
        if all([p in allPaths for p in pathsToSave]):
            pathsToDelete = [p for p in allPaths if not p in pathsToSave]
        else:
            raise Exception(f"One of paths {pathsToSave} not exists")
        print(f"Saves: {', '.join(pathsToSave)}")
        print("======= REMOVE PATHS")
        for currentPath in pathsToDelete:
            if os.path.exists(os.path.join(currentPath, "files")):
                shutil.rmtree(os.path.join(currentPath, "files")) #remove only files path

        endTime = datetime.now()
        print("=======")
        print(endTime.strftime(timeFormat))
        print("====================")
        # notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()), "Start:\t{}\nEnd:\t{}\nDuration:\t{}".format(startTime.strftime(timeFormat),endTime.strftime(timeFormat),str(endTime-startTime).split('.', 2)[0]))

