import numpy as np


def diff(arr1, arr2):
    return [item for item in arr1 if item not in set(arr2)]


def combine_unique(arr1, arr2):
    return list(set(np.append(arr1, arr2)))
