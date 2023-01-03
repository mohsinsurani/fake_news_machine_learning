import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support as score, roc_curve, \
    roc_auc_score, mean_squared_error, r2_score, classification_report
import seaborn as sns
from matplotlib import pyplot as plt


class PlotCf:

    @staticmethod
    def draw_res(y_test, y_pred):
        cf = confusion_matrix(y_test, y_pred)
        print(classification_report(y_test, y_pred))

        cf_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        cf_counts = ['{0:0.0f}'.format(val) for val in
                     cf.flatten()]
        cf_perc = ['{0:.2%}'.format(val) for val in
                   cf.flatten() / np.sum(cf)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                  zip(cf_names, cf_counts, cf_perc)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf, annot=labels, cmap="viridis", fmt='')
        plt.figure(figsize=(10, 7))

        precision, recall, fscore, support = score(y_test, y_pred)

        print(accuracy_score(y_pred, y_test), "Accuracy")

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

        PlotCf.draw_roc(y_test, y_pred)

    @staticmethod
    def draw_roc(y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print('AUC: %.3f' % auc)
        print('Mean Squared Error: ', mean_squared_error(y_true=y_test, y_pred=y_pred))
        print('Coefficient of determination: %.2f'
              % r2_score(y_true=y_test, y_pred=y_pred))

        print(metrics.roc_auc_score(y_test, y_pred))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.plot(fpr, tpr, label="auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()
