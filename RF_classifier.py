import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import lm
from sklearn import metrics, model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support as score, \
    classification_report, roc_curve, roc_auc_score, mean_squared_error, r2_score

from preprocess import Preproc

preprocess = Preproc()
df = preprocess.fetch_df_diff()

X = preprocess.get_df(df=df)
y = df['target']
X_train, X_test, y_train, y_test = preprocess.get_train_test_countVector(X, y)
model = Preproc.get_rf_model(X_train, y_train)
X_train_prediction = model.predict(X_train)
print(X_train_prediction)

training_accuracy = accuracy_score(X_train_prediction, y_train)
print(training_accuracy)

X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, y_test)
print(testing_accuracy)

cf = confusion_matrix(y_test, X_test_prediction)

cf_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
cf_counts = ['{0:0.0f}'.format(val) for val in
             cf.flatten()]
cf_perc = ['{0:.2%}'.format(val) for val in
           cf.flatten() / np.sum(cf)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(cf_names, cf_counts, cf_perc)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf, annot=labels, cmap="viridis", fmt='')

precision, recall, fscore, support = score(y_test, X_test_prediction)
print(accuracy_score(X_test_prediction, y_test), "acc")

fpr, tpr, thresholds = roc_curve(y_test, X_test_prediction)
auc = roc_auc_score(y_test, X_test_prediction)
print('AUC: %.3f' % auc)
print('Mean Squared Error: ', mean_squared_error(y_true=y_test, y_pred=X_test_prediction))
print('Coefficient of determination: %.2f'
      % r2_score(y_true=y_test, y_pred=X_test_prediction))

print(metrics.roc_auc_score(y_test, X_test_prediction))

plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


def Statistics(y_test, X_test_prediction):
    # Classification Report
    print("Classification Report is shown below")
    print(classification_report(y_test, X_test_prediction))
    cm = confusion_matrix(y, X_test_prediction)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')


Statistics(y_test, X_test_prediction)
