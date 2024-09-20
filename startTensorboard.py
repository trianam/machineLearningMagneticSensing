#!/usr/bin/env python

import sys
import os
import configurations
from ray.tune import ExperimentAnalysis

if len(sys.argv) < 2 or len(sys.argv) > 2:
    print("Use {} configName".format(sys.argv[0]))
else:
    # conf = getattr(sys.modules['configurations'], sys.argv[1])
    conf = eval('configurations.{}'.format(sys.argv[1]))
    print("====================")
    print("RUN USING {}".format(sys.argv[1]))
    if conf.useTune:
        mode = ("max" if conf.bestSign == '>' else "min")
        analysis = ExperimentAnalysis(os.path.abspath(os.path.join("tuneOutput", conf.path)))
        logDir = analysis.get_best_logdir(metric="/".join(["valid",conf.bestKey]), mode=mode)
    else:
        logDir = f"files/{conf.path}/tensorBoard"
    print(f"tensorboard --logdir={logDir}")
    print("====================")
    os.system(f"tensorboard --logdir={logDir}")