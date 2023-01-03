import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support as score, \
    classification_report
import nltk
nltk.download('stopwords')
stemmer = PorterStemmer()

def fetch_df():
    fake_df = pd.read_csv('/content/drive/MyDrive/fake_news_project/archive/Fake.csv')
    true_df = pd.read_csv('/content/drive/MyDrive/fake_news_project/archive/True.csv')

    true_df['target'] = 1
    fake_df['target'] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True, sort=False)
    df = del_unused(df)
    return df

def get_train_test_countVector(X, y):
    X = get_vector_count(X)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test



def get_train_test_tfidVector(X, y):
    X = get_vector_tfidf(X)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2)
    return X_train, X_test, y_train, y_test



def del_unused(df):
    df['text'] = df['title'] + ' ' + df['text']
    del df['title']
    del df['subject']
    del df['date']
    return df

def clean_data(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # removes everything that is not a letter
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def get_df(df):
    filename = 'df_x.sav'
    path = Path(filename)
    if path.is_file():
        df['text'] = pickle.load(open(filename, 'rb'))
        return df['text']
    else:
        df['text'] = df['text'].apply(clean_data)
        df['text'].head()
        pickle.dump(df['text'], open(filename, 'wb'))
        return df['text']

def get_vector_tfidf(X):
    filename = 'vector_x.sav'
    path = Path(filename)
    if path.is_file():
        X = pickle.load(open(filename, 'rb'))
        return X
    else:
        X = TfidfVectorizer().fit_transform(X)
        pickle.dump(X, open(filename, 'wb'))
        return X


def get_vector_count(X):
    filename = 'vector_count_x.sav'
    path = Path(filename)
    if path.is_file():
        X = pickle.load(open(filename, 'rb'))
        return X
    else:
        vectorizer = CountVectorizer()
        vectorizer.fit(X)

        print(vectorizer.vocabulary_)

        X = vectorizer.transform(X)
        pickle.dump(X, open(filename, 'wb'))
        return X


def getModel(X_train, y_train):
    filename = 'log_reg_model.sav'
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


def getSVMModel(X_train, y_train):
    filename = 'svm_model.sav'
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        model = svm.SVC(kernel='linear')  # Linear Kernel
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


def getGBModel(X_train, y_train):
    filename = 'gb_model.sav'
    path = Path(filename)
    if path.is_file():
        model = pickle.load(open(filename, 'rb'))
        return model
    else:
        model = GaussianNB()
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb'))
        return model


df = fetch_df()
X = get_df(df=df)
y = df['target']
X_train, X_test, y_train, y_test = get_train_test_countVector(X, y)
model = getSVMModel(X_train, y_train)
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

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


def Statistics(y_test, X_test_prediction):
    # Classification Report
    print("Classification Report is shown below")
    print(classification_report(y_test, X_test_prediction))

    # Confusion matrix
    print("Confusion matrix is shown below")
    cm = confusion_matrix(y_test, X_test_prediction)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')


Statistics(y_test, X_test_prediction)
