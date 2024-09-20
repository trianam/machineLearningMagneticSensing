#!/usr/bin/env python
# coding: utf-8

#     calculateTuneTestMetrics.py
#     Reports all the results after the training for the best models in the considered categories.
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

from funPlot import printMetricsOld
from funPlot import printMetrics
from itertools import product
import numpy as np

# printMetrics('trun1')
# printMetrics('trun2')
printMetrics('fgsfrun32ct')
