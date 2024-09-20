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

from funPlot import plotError
from itertools import product


# plotError(['files/fgsfrun13/predictions/best-97-real.npz', 'files/fgsfrun13s2/predictions/best-81-real.npz', 'files/fgsfrun13s3/predictions/best-82-real.npz', 'files/fgsfrun13s4/predictions/best-72-real.npz', 'files/fgsfrun13s5/predictions/best-65-real.npz', 'files/fgsfrun13s6/predictions/best-79-real.npz', 'files/fgsfrun13s7/predictions/best-69-real.npz', 'files/fgsfrun13s8/predictions/best-72-real.npz'], ylim=(0,3), label="Real sample", save="gsfrun13Real")
# plotError(['files/fgsfrun13/predictions/last-199-test.npz', 'files/fgsfrun13s2/predictions/last-199-test.npz', 'files/fgsfrun13s3/predictions/last-199-test.npz', 'files/fgsfrun13s4/predictions/last-199-test.npz', 'files/fgsfrun13s5/predictions/last-199-test.npz', 'files/fgsfrun13s6/predictions/last-199-test.npz', 'files/fgsfrun13s7/predictions/last-199-test.npz', 'files/fgsfrun13s8/predictions/last-199-test.npz'], ylim=(0,3), label="Synthetic set", save="gsfrun13Synth")

plotError([
    ['files/fgsfrun13/predictions/best-97-real.npz', 'files/fgsfrun13s2/predictions/best-81-real.npz', 'files/fgsfrun13s3/predictions/best-82-real.npz', 'files/fgsfrun13s4/predictions/best-72-real.npz', 'files/fgsfrun13s5/predictions/best-65-real.npz', 'files/fgsfrun13s6/predictions/best-79-real.npz', 'files/fgsfrun13s7/predictions/best-69-real.npz', 'files/fgsfrun13s8/predictions/best-72-real.npz'],
    ['files/fgsfrun13/predictions/last-199-test.npz', 'files/fgsfrun13s2/predictions/last-199-test.npz', 'files/fgsfrun13s3/predictions/last-199-test.npz', 'files/fgsfrun13s4/predictions/last-199-test.npz', 'files/fgsfrun13s5/predictions/last-199-test.npz', 'files/fgsfrun13s6/predictions/last-199-test.npz', 'files/fgsfrun13s7/predictions/last-199-test.npz', 'files/fgsfrun13s8/predictions/last-199-test.npz'],
    ], ylim=(0,3), label=["Real sample", "Synthetic set"], save="gsfrun13")