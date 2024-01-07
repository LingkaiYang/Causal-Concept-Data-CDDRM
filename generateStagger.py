import numpy as np
import pandas as pd
from river.datasets import synth

def generate_data_batch(n_sample, tag):
    # generate a data batch that has self.n_sample samples, return pd.DataFrame including both features and labels
    dataset = synth.STAGGER(classification_function=tag)
    X, Y = [], []
    for x, y in dataset.take(n_sample):
        X.append(list(x.values()))
        Y.append(y)
    arr = np.concatenate((np.array(X), np.array(Y).reshape(-1, 1)), axis=1)
    df_XY = pd.DataFrame(arr)
    return df_XY

def generate_drift_data(n_sample, tag1, tag2, n_drift_points):
    # generate a data series that has self.n_sample*self.n_drift_points samples,
    # return pd.DataFrame including both features and labels
    df = pd.DataFrame()
    for _ in np.arange(int(n_drift_points * 0.5)):
        new_df0 = generate_data_batch(n_sample, tag1)
        new_df1 = generate_data_batch(n_sample, tag2)
        df = pd.concat([df, new_df0, new_df1], ignore_index=True)
    return df


n_sample = 5000
w_size = 2000
stride = 2000
n_drift_points = 10

for args in [[0,1], [0,2], [1,0], [1,2], [2,0], [2,1]]:
    tag1, tag2 = args
    # df is generated Stagger data with (n_drift_points-1) concept drift points between Mode A and Mode B in args.
    df = generate_drift_data(n_sample, tag1, tag2, n_drift_points)

