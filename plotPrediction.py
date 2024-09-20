import argparse
import numpy as np
import matplotlib.pyplot as plt

import configurations
import os
import funPytorch as fun
import loaders


datasets = {
    'datasetA4HF1P': {
        'datasetFilename': 'datasets/datasetA4HF1P.npz',
        'set': 'valid',
        'dataIndex': 0,
        'modelToUse': 'fgsfrun20t',
        'savePlot': 'fgsfrun20t',
    },
}

for s in range(2, 41):
    datasets[f'datasetA4HF1Ps{s}'] = {
        **datasets[f'datasetA4HF1P'],
        'datasetFilename': f'datasets/datasetA4HF1Ps{s}.npz',
        'modelToUse': f'fgsfrun20ts{s}',
        'savePlot': f'fgsfrun20ts{s}',
    }

for s in list(range(2,26))+[27,28,29,30,32,34,36,38,40]: #to avoid doubles with same number of measurements
    datasets[f'datasetIA4HF1Pts{s}'] = {
        'datasetFilename': f'datasets/datasetIA4HF1Ps{s}.npz',
        'set': 'valid',
        'dataIndex': 0,
        'modelToUse': f'fgsfrun23ts{s}',
        'savePlot': f'fgsfrun23ts{s}',
    }
    datasets[f'datasetIA4HF1Ps{s}'] = {
        'datasetFilename': f'datasets/datasetIA4HF1Ps{s}.npz',
        'set': 'valid',
        'dataIndex': 0,
        'modelToUse': f'fgsfrun23s{s}',
        'savePlot': f'fgsfrun23s{s}',
    }

def main(confName):
    data = datasets[confName]

    d = np.load(data['datasetFilename'])
    freq = d['freq']
    x = d[f"{data['set']}X"][data['dataIndex']]
    y = d[f"{data['set']}Y"][data['dataIndex']]



    conf = eval('configurations.{}'.format(data['modelToUse']))
    conf.runningPredictions = True
    # device = "cuda:0"
    device = "cpu"

    print("======= LOAD MODEL")
    model, optim, loadEpoch, _ = fun.loadModel(conf, device)
    print("======= LOAD DATA")
    dataloaders,_ = loaders.custom(conf, x.reshape(1,-1), y.reshape(1,-1), batchSize=1, shuffleDataset=False)
    print("======= CALCULATE PREDICTIONS")
    preds = fun.predict(conf, model, dataloaders, loadEpoch, toSave=False, toReturn=True)

    plt.title(f"{confName}\npeaks:[{', '.join([f'{p:.2f}' for p in y])}]\npreds:[{', '.join([f'{p:.2f}' for p in preds['custom']['pred'][0]])}]\n(MAE {sum([abs(t - p) for t, p in zip(y, preds['custom']['pred'][0])]) / len(y):.2f})", fontsize=8)
    plt.xlabel("MHz")

    plt.plot(freq, x, label="Meas", marker='.', linestyle='None')
    plt.vlines(y, 0, np.max(x), colors="C0")
    plt.vlines(preds['custom']['pred'][0], 0, np.max(x), colors="C1", linestyles='dashed',
               label="Pred")

    plt.legend()
    if not data['savePlot'] is None:
        if not os.path.exists("plots"):
            os.makedirs("plots")
        saveTo = f"plots/plot-pred-{data['savePlot']}.pdf"
        plt.savefig(saveTo, bbox_inches='tight')
        print(f"Saved to: {saveTo}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    parser.add_argument('configName', type=str, help='Dataset configuration name')

    args = parser.parse_args()
    main(args.configName)