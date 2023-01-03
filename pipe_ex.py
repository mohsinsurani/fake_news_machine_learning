# Scale the data
import pickle
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
# Pipeline, Gridsearch, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
# Plot the confusion matrix at the end of the tutorial
from sklearn.metrics import plot_confusion_matrix
# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import svm
from preprocess import Preproc

preprocess = Preproc()
df = preprocess.fetch_df_diff()

X = preprocess.get_df(df=df)
y = df['target']
X_train, X_test, y_train, y_test = preprocess.get_train_test_countVector(X, y)


pipe_rf = Pipeline([('scl', StandardScaler(with_mean=False)),
                    ('RF', RandomForestClassifier(random_state=42))])
pipe_xgb = Pipeline([('scl', StandardScaler(with_mean=False)),
                     ('XGB', XGBClassifier(random_state=42))])

param_range = [1, 2, 3, 4, 5, 6]
param_range_fl = [1.0, 0.5, 0.1]
n_estimators = [50, 100, 150]
learning_rates = [.1, .2, .3]

rf_param_grid = [{'RF__min_samples_leaf': param_range,
                  'RF__max_depth': param_range,
                  'RF__min_samples_split': param_range[1:]}]


xgb_param_grid = [{'XGB__learning_rate': learning_rates,
                   'XGB__max_depth': param_range,
                   'XGB__min_child_weight': param_range[:2],
                   'XGB__subsample': param_range_fl,
                   'XGB__n_estimators': n_estimators}]

gs_param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

rf_grid_search = GridSearchCV(estimator=pipe_rf,
                              param_grid=rf_param_grid,
                              scoring='accuracy',
                              cv=3)

xgb_grid_search = GridSearchCV(estimator=pipe_xgb,
                               param_grid=xgb_param_grid,
                               scoring='accuracy',
                               cv=3)


grid_dict = {0: 'Random Forest',
             1: 'XGBoost'}

grids = [rf_grid_search, xgb_grid_search]
for i, pipe in enumerate(grids):
    filename = grid_dict[i]
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
    else:
        model = pipe.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))

    print('{} Test Accuracy: {}'.format(grid_dict[i],
                                        model.score(X_test, y_test)))
    print('{} Best Params: {}'.format(grid_dict[i], model.best_params_))
