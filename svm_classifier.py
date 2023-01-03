import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from plotCF import PlotCf
from preprocess import Preproc

isTFID = True
isGrid = True

preprocess = Preproc()
df = preprocess.fetch_df_diff()

X = preprocess.get_df(df=df)
y = df['target']


def fetch_model(X_train, y_train, isGrid):
    if isGrid:
        return getGridModel(X_train, y_train)
    else:
        return getSVMModel(X_train, y_train)


def getSVMModel(X_train, y_train):
    filename = 'svm_model_{}.sav'.format(isTFID)
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        model = SVC(random_state=2)
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


def getGridModel(X_train, y_train):  # 91
    filename = 'svm_grid_model_{}.sav'.format(isTFID)
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        param_grid = {'C': [0.1, 1, 10],
                      'gamma': [1, 0.1, 0.01],
                      'kernel': ['linear', 'rbf']}

        # param_grid = {'C': [10],
        #               'gamma': [1],
        #               'kernel': ['rbf']}

        model = GridSearchCV(SVC(cache_size=200), param_grid, refit=True, verbose=3)
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


X_train, X_test, y_train, y_test = preprocess.get_train_test(X, y, isTFID=isTFID)
model = fetch_model(X_train, y_train, isGrid=isGrid)
X_train_prediction = model.predict(X_train)
print(X_train_prediction)

training_accuracy = accuracy_score(X_train_prediction, y_train)
print(training_accuracy)

X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, y_test)
print(testing_accuracy)

if isGrid:
    print("best model param", model.best_params_)

PlotCf.draw_res(y_test, X_test_prediction)
