import os
import numpy as np
import argparse
import matlab.engine
import matplotlib.pyplot as plt


configurations = {
    'FGHF1P': {
        'simScript': 'sim/mock_diamond2_new_galya.m',
        'N': 600,
        'center_freq': 2890,
        'half_window_size': 360,
        'noiseSigma': 0, #0.00055, #(0.0002, 0.001), # 0.0002725640441769875 - 0.0009385689629442084
        'B_mag': 80, #(80, 120), # 808.079662936 - 1135.134239139
        'B_theta': 57, #(57, 73), # 57.65151589 - 72.01018995
        'B_phi': 54, #(54, 80), # 54.01828013 - 79.26266928
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': 1,
        'saveFilename': None
    },
}

for s in range(2,41):
    configurations[f'FGHF1Ps{s}'] = {
        'simScript': 'sim/mock_diamond2_new_galya.m',
        'N': 600,
        'center_freq': 2890,
        'half_window_size': 360,
        'noiseSigma': 0.00055, #(0.0002, 0.001),  # 0.0002725640441769875 - 0.0009385689629442084
        'B_mag': 80, #(80, 120),  # 808.079662936 - 1135.134239139
        'B_theta': 57, #(57, 73),  # 57.65151589 - 72.01018995
        'B_phi': 54, #(54, 80),  # 54.01828013 - 79.26266928
        'fDataName': 'smp_freqs',
        'xDataName': 'sig',
        'yDataName': 'peak_locs',
        'takeEvery': s,
        'saveFilename': None
    }


def main(confName):
    conf = configurations[confName]
    workspace = os.path.dirname(conf['simScript'])
    funName = os.path.splitext(os.path.basename(conf['simScript']))[0]

    eng = matlab.engine.start_matlab()
    simPath = eng.genpath(workspace)
    eng.addpath(simPath, nargout=0)

    N = conf['N']
    center_freq = conf['center_freq']
    half_window_size = conf['half_window_size']


    noiseSigma = np.random.uniform(conf['noiseSigma'][0], conf['noiseSigma'][1]) if isinstance(conf['noiseSigma'], (list,tuple)) else conf['noiseSigma']
    B_mag = np.random.uniform(conf['B_mag'][0], conf['B_mag'][1]) if isinstance(conf['B_mag'], (list,tuple)) else conf['B_mag']
    B_theta = np.random.uniform(conf['B_theta'][0], conf['B_theta'][1]) if isinstance(conf['B_theta'], (list, tuple)) else conf['B_theta']
    B_phi = np.random.uniform(conf['B_phi'][0], conf['B_phi'][1]) if isinstance(conf['B_phi'], (list, tuple)) else conf['B_phi']
    diamond = eval(
        f"eng.{funName}(float(N), float(center_freq), float(half_window_size), float(noiseSigma), float(B_mag), float(B_theta), float(B_phi))")
    eng.workspace["wDiamond"] = diamond

    freq = np.array(eng.eval(f"wDiamond.{conf['fDataName']}")).reshape(-1)[::conf['takeEvery']]
    X = np.array(eng.eval(f"wDiamond.{conf['xDataName']}")).reshape(-1)[::conf['takeEvery']]
    y = np.array(eng.eval(f"wDiamond.{conf['yDataName']}")).reshape(-1)

    plt.vlines(y, 0, np.max(X), colors="C1")
    plt.plot(freq, X)
    plt.xlabel("MHz")
    plt.title(
        f"{confName}\n" 
        rf"$\sigma={noiseSigma};\quad B_{{mag}}={B_mag};\quad B_\theta={B_theta};\quad B_\varphi={B_phi}$"
        "\n" 
        rf"$B=[{', '.join([f'{v:.2f}' for v in y])}]$", fontsize=8)
    if not conf['saveFilename'] is None:
        plt.savefig(conf['saveFilename'], bbox_inches='tight')
        print(f"Saved to: {conf['saveFilename']}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)
