import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        return getLogModel(X_train, y_train)


def getLogModel(X_train, y_train):
    filename = 'log_reg_model_{}.sav'.format(isTFID)
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        model = LogisticRegression(random_state=2)
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


def getGridModel(X_train, y_train):  # 91
    filename = 'log_reg_grid_model_{}.sav'.format(isTFID)
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        param_range_fl = [1.0, 0.5, 0.1]
        lr_param_grid = [{'LR__penalty': ['l1', 'l2'],
                          'LR__C': param_range_fl,
                          'LR__solver': ['liblinear', 'sag']}]
        pipe_lr = Pipeline([('scl', StandardScaler(with_mean=False)),
                            ('LR', LogisticRegression(random_state=42))])
        model = GridSearchCV(estimator=pipe_lr,
                             param_grid=lr_param_grid,
                             scoring='accuracy',
                             cv=3)
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
