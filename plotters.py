import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report


# feat hist and feat dispersion hist
def plot_feat_scatter(df, feat_name, target_name):
    fig, ax = plt.subplots(1, 2)
    ax1, ax2 = ax.flatten()

    print(f'{feat_name}:')

    ax2.scatter(df[target_name], df[feat_name])
    ax1.hist(df[feat_name])

    fig.set_size_inches(12, 3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.show()


# feats hist and feat dispersion hist
def plot_feat_scatter_all(df, features, target_name):
    for col_name in features:
        plot_feat_scatter(df, col_name, target_name)


# feats to target correlation
def plot_corr(df, features, target_name):
    corr_with_target = df[np.concatenate((features, [target_name]))].corr().iloc[:-1, -1].sort_values(ascending=False)

    plt.figure(figsize=(10, 8))

    sns.barplot(x=corr_with_target.values, y=corr_with_target.index)

    plt.title('Correlation with target variable')
    plt.show()


def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred):
    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))
    print('TEST\n\n' + classification_report(y_test_true, y_test_pred))
    print('CONFUSION MATRIX\n')
    print(pd.crosstab(y_test_true, y_test_pred))


