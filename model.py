import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC


def train_Model_and_apply(train_features: np.array, train_labels: np.array, test_features: np.array, modeltype: str = 'RF') -> np.array:
    if modeltype == 'RF':
        model = RandomForestRegressor().fit(X=train_features, y=train_labels.ravel())
    elif modeltype == 'SVC':
        model = SVC(kernel='linear').fit(X=train_features, y=train_labels.ravel())
    else:
        raise NameError('Modeltype is not implemented!')
        return -1

    result = model.predict(X=test_features)
    return result


def train_RF_and_apply(train_features: np.array, train_labels: np.array, test_features: np.array) -> np.array:
    forest = RandomForestRegressor().fit(X=train_features, y=train_labels.ravel())
    result = forest.predict(X=test_features)
    return result
