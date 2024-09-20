import numpy as np
import argparse
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

configurations = {
    'run1': {
        'dataset': 'datasets/dataset1.npz',
        'polyDegree': None,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run1.npz',
    },
    'run2': {
        'dataset': 'datasets/dataset1.npz',
        'polyDegree': 2,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run2.npz',
    },
    'run3': {
        'dataset': 'datasets/dataset1.npz',
        'polyDegree': None,
        'regularization': 'lasso',
        'alpha': 1.0,
        'predictions': 'predictionsLinearReg/run3.npz',
    },
    'run4': {
        'dataset': 'datasets/dataset1.npz',
        'polyDegree': None,
        'regularization': 'ridge',
        'alpha': 1.0,
        'predictions': 'predictionsLinearReg/run4.npz',
    },
    'run5': {
        'dataset': 'datasets/dataset2.npz',
        'polyDegree': None,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run5.npz',
    },
    'run5b': {
        'dataset': 'datasets/dataset2.npz',
        'polyDegree': 2,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run5b.npz',
    },
    'run6': {
        'dataset': 'datasets/dataset3.npz',
        'polyDegree': None,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run6.npz',
    },
    'run6b': {
        'dataset': 'datasets/dataset3.npz',
        'polyDegree': 2,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run6b.npz',
    },
    'run7': {
        'dataset': 'datasets/dataset4.npz',
        'polyDegree': None,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run7.npz',
    },
    'run7b': {
        'dataset': 'datasets/dataset4.npz',
        'polyDegree': 2,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run7b.npz',
    },
    'run8': {
        'dataset': 'datasets/dataset5.npz',
        'polyDegree': None,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run8.npz',
    },
    'run8b': {
        'dataset': 'datasets/dataset5.npz',
        'polyDegree': 2,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run8b.npz',
    },
    'run9': {
        'dataset': 'datasets/dataset6.npz',
        'polyDegree': None,
        'regularization': None,
        'predictions': 'predictionsLinearReg/run9.npz',
    },
    'frun1': {
        'dataset': 'datasets/dataset2f.npz',
        'polyDegree': None,
        'regularization': None,
        'predictions': 'predictionsLinearReg/frun1.npz',
    },
}

def main(confName):
    myConf = configurations[confName]

    print("Load data")
    dataset = np.load(myConf['dataset'])

    if not myConf['polyDegree'] is None:
        if not myConf['regularization'] is None:
            if myConf['regularization'] == 'lasso':
                reg = Pipeline([
                    ('poly', PolynomialFeatures(degree=myConf['polyDegree'])),
                    ('lasso', Lasso(alpha=myConf['alpha'], fit_intercept=False))])
            elif myConf['regularization'] == 'ridge':
                reg = Pipeline([
                    ('poly', PolynomialFeatures(degree=myConf['polyDegree'])),
                    ('ridge', Ridge(alpha=myConf['alpha'], fit_intercept=False))])
            else:
                raise ValueError(f"Regularization {myConf['regularization']} not valid.")
        else:
            reg = Pipeline([
                ('poly', PolynomialFeatures(degree=myConf['polyDegree'])),
                ('linear', LinearRegression(fit_intercept=False))])
    else:
        if not myConf['regularization'] is None:
            if myConf['regularization'] == 'lasso':
                reg = Lasso(alpha=myConf['alpha'])
            elif myConf['regularization'] == 'ridge':
                reg = Ridge(alpha=myConf['alpha'])
            else:
                raise ValueError(f"Regularization {myConf['regularization']} not valid.")
        else:
            reg = LinearRegression()

    print("Fit regressor")
    # reg.fit(dataset['trainX'], dataset['trainY'].reshape([-1])) #adjust reshape when multiple regression
    reg.fit(dataset['trainX'], dataset['trainY'])

    if not myConf['polyDegree'] is None:
        inputFeatures = reg[-1].n_features_in_
    else:
        inputFeatures = reg.n_features_in_

    print(f"Input features: {inputFeatures}")
    print(f"Output dimension: {dataset['trainY'].shape[1]}")
    splits = ['train','valid','test']

    print("Calculate predictions")
    pred = {s: reg.predict(dataset[f'{s}X']) for s in splits}

    print("Calculate MAE")
    # mae = {s: np.average(np.abs(dataset[f'{s}Y'].reshape([-1]) - pred[s])) for s in splits} #adjust reshape when multiple regression
    mae = {s: np.average(np.abs(dataset[f'{s}Y'] - pred[s])) for s in splits}
    for s in splits:
        print(f'{s} MAE: {mae[s]}')

    if dataset['trainY'].shape[1] > 1:
        maes = {s: np.average(np.abs(dataset[f'{s}Y'] - pred[s]),axis=0) for s in splits}
        for s in splits:
            print(f'{s} MAEs: {maes[s]}')

    print("Saving predictions")
    np.savez_compressed(myConf['predictions'], trainP=pred['train'], validP=pred['valid'], testP=pred['test'])
    print(f"Saved predictions in {myConf['predictions']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=52,width=90))
    parser.add_argument('configName', type=str, help='Run configuration name')

    args = parser.parse_args()
    main(args.configName)
