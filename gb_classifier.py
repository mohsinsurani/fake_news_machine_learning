import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from plotCF import PlotCf
from preprocess import Preproc

isTFID = False
isGrid = False
isMulti = True
preprocess = Preproc()
df = preprocess.fetch_df_diff()

X = preprocess.get_df(df=df)
y = df['target']


def fetch_model(X_train, y_train, isGrid, isMulti):
    if isMulti:
        return getMultiGaussModel(X_train, y_train)
    elif isGrid:
        return getGridModel(X_train, y_train)
    else:
        return getGaussModel(X_train, y_train)

def getMultiGaussModel(X_train, y_train):
    filename = 'gb_multimodel_{}.sav'.format(isTFID)
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        model = MultinomialNB()
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model

def getGaussModel(X_train, y_train):
    filename = 'gb_model_{}.sav'.format(isTFID)
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        model = GaussianNB()
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


def getGridModel(X_train, y_train):  # 91
    filename = 'gb_grid_model_{}.sav'.format(isTFID)
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        gs_param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
        model = GridSearchCV(GaussianNB(), gs_param_grid, refit=True, verbose=3)
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


X_train, X_test, y_train, y_test = preprocess.get_train_test(X, y, isTFID=isTFID)
model = fetch_model(X_train.todense(), y_train, isGrid=isGrid, isMulti=isMulti)
X_train_prediction = model.predict(X_train.todense())
print(X_train_prediction)

training_accuracy = accuracy_score(X_train_prediction, y_train)
print(training_accuracy)

X_test_prediction = model.predict(X_test.todense())
testing_accuracy = accuracy_score(X_test_prediction, y_test)
print(testing_accuracy)

if isGrid:
    print("best model param", model.best_params_)

PlotCf.draw_res(y_test, X_test_prediction)
