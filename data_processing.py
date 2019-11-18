import numpy as np
from sklearn.model_selection import train_test_split


# DATA PREPARATION
def normalize(df, features):
    df2 = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
    return df2.combine_first(df)


# FEATURE ENGINEERING
def new_feat_log(df, features):
    pass


def new_feats_numeric(df, features):
    new_feats = np.array([])

    for feat in features:
        if np.issubdtype(df[feat].dtype, np.number):
            df[f'sqr_{feat}'] = df[feat] ** 2
            df[f'sqrt_{feat}'] = df[feat] ** 0.5
            df[f'log_{feat}'] = df[feat].apply(lambda x: np.log(x) if x > 1 else 10 ** -6)

            new_feats = np.append([f'sqr_{feat}', f'sqrt_{feat}', f'log_{feat}'], new_feats)

    return df, new_feats


def split(df, features, target_name, test_size=0.25):
    X = df[features]
    y = df[target_name]

    return train_test_split(X, y, test_size=test_size, random_state=42)
