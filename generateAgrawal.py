import numpy as np
import pandas as pd
from river.datasets import synth

def generate_data_batch(n_sample, tag):
    # generate a data batch that has self.n_sample samples, return pd.DataFrame including both features and labels
    dataset = synth.Agrawal(classification_function=tag)
    X, Y = [], []
    for x, y in dataset.take(n_sample):
        X.append(list(x.values()))
        Y.append(y)
    arr = np.concatenate((np.array(X), np.array(Y).reshape(-1, 1)), axis=1)
    df_XY = pd.DataFrame(arr)
    df_XY = define_base_labels(df_XY)
    return df_XY

def generate_drift_data(n_sample, tag, n_drift_points, type_drift, args):
    # generate a data series that has self.n_sample*self.n_drift_points samples,
    # return pd.DataFrame including both features and labels
    df = pd.DataFrame()
    for _ in np.arange(int(n_drift_points * 0.5)):
        new_df0 = generate_data_batch(n_sample, tag)
        df1 = generate_data_batch(n_sample, tag)
        if type_drift == 1:     #  features unchanged, change the rule for classification labels, i.e., real drift
            new_df1 = define_drift_labels(df1, args[0], args[1])
        if type_drift == 2:     #  features changed, keep the rule for classification labels, i.e., virtual drift
            new_df1 = define_drift_features(df1, args[0], args[1])
        df = pd.concat([df, new_df0, new_df1], ignore_index=True)
    return df

def define_base_labels(df):
    # based on salary, commission, loan and hvalue
    features = df.values[:,0:-1]
    disposable = 0.67 * (features[:,0] + features[:,1]) - 0.2 * features[:,8] + 0.1 * features[:,6]
    labels = [1 if i > 0 else 0 for i in disposable]
    df[9] = labels
    return df

def define_drift_labels(df, coe1, coe2):
    # based on salary, commission, loan and hvalue
    features = df.values[:,0:-1]
    disposable = 0.1 * features[:,0] - coe1 * features[:,8] + coe2 * features[:,6]
    labels = [1 if i > 0 else 0 for i in disposable]
    df[9] = labels
    return df

def define_drift_features(df, coe1, coe2):
    # to generate virtual concept drift by increasing the value of a number of features
    # based on salary, commission, loan and hvalue
    features = df.values[:,coe1]
    scale = np.array([np.random.uniform(coe2[0], coe2[1]) for _ in np.arange(len(features.flatten()))]).reshape(features.shape)
    new_features = features * scale
    df[coe1] = new_features
    return df

n_sample = 5000
w_size = 2000
stride = 2000
n_drift_points = 10

# generating real concept drift
tag = 1
type_drift = 1
for coe1 in [0.1, 0.3]:
    for coe2 in [0.3, 0.5]:
        args = [coe1, coe2]
        df = generate_drift_data(n_sample, tag, n_drift_points, type_drift, args)

# generating virtual concept drift
tag = 1
type_drift = 2
temp = np.arange(0, 2.1, 0.2)
vary_range = [[np.round(i, decimals=1), np.round(j, decimals=1)] for i, j in zip(temp[0:-1], temp[1:])]
vary_features = [np.arange(i).tolist() for i in [1, 3, 5, 7, 9]]
for coe1 in vary_features:
    for coe2 in vary_range:
        args = [coe1, coe2]
        df = generate_drift_data(n_sample, tag, n_drift_points, type_drift, args)