import pickle
import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB

stemmer = PorterStemmer()


class Preproc:

    def fetch_df_diff(self):
        filename = 'shuffled_df_diff.sav'
        path = Path(filename)
        if path.is_file():
            df = pickle.load(open(filename, 'rb'))
            return df
        else:
            df = pd.read_csv('archive/news.csv')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df['target'] = df['label'].map({'REAL': 1, 'FAKE': 0})
            df['text'] = df['title'] + ' ' + df['text']
            del df['title']
            del df['label']
            df.head()
            pickle.dump(df, open(filename, 'wb'))
            return df

    @staticmethod
    def get_train_test(X, y, isTFID):
        if isTFID:
            return Preproc.get_train_test_tfidVector(X, y)
        else:
            return Preproc.get_train_test_countVector(X, y)

    @staticmethod
    def get_train_test_countVector(X, y):
        X = Preproc.get_vector_count(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_train_test_tfidVector(X, y):
        X = Preproc.get_vector_tfidf(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2)
        return X_train, X_test, y_train, y_test

    def clean_data(self, content):
        text = re.sub('[^a-zA-Z]', ' ', content)
        text = text.lower()
        text = text.split()
        text = [stemmer.stem(word) for word in text if not word in stopwords.words('english')]
        text = ' '.join(text)
        return text

    def get_df(self, df):
        filename = 'df_x.sav'
        path = Path(filename)
        if path.is_file():
            df['text'] = pickle.load(open(filename, 'rb'))
            df['text'].head()
            return df['text']
        else:
            df['text'] = df['text'].apply(self.clean_data)
            df['text'].head()
            pickle.dump(df['text'], open(filename, 'wb'))
            return df['text']

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def getMultinomialNBModel(X_train, y_train):  # 89
        filename = 'MultinomialNB_gauss_model.sav'
        path = Path(filename)
        if path.is_file():
            model = pickle.load(open(filename, 'rb'))
            return model
        else:
            model = MultinomialNB()
            model.fit(X_train, y_train)
            pickle.dump(model, open(filename, 'wb'))
            return model

    @staticmethod
    def get_rf_model(X_train, y_train):  # 82
        filename = 'rf_model.sav'
        path = Path(filename)
        if path.is_file():
            model = pickle.load(open(filename, 'rb'))
            return model
        else:
            param_range = [1, 2, 3, 4, 5, 6]
            rf_param_grid = [{'RF__min_samples_leaf': param_range,
                              'RF__max_depth': param_range,
                              'RF__min_samples_split': param_range[1:]}]
            model = GridSearchCV(RandomForestClassifier(random_state=42),
                                 param_grid=rf_param_grid,
                                 scoring='accuracy',
                                 cv=3)
            model.fit(X_train, y_train)
            pickle.dump(model, open(filename, 'wb'))
            return model



    # def fetch_df(self):
    #     filename = 'shuffled_df.sav'
    #     path = Path(filename)
    #     if path.is_file():
    #         df = pickle.load(open(filename, 'rb'))
    #         return df
    #     else:
    #         fake_df = pd.read_csv('archive/Fake.csv')
    #         true_df = pd.read_csv('archive/True.csv')
    #
    #         true_df['target'] = 1
    #         fake_df['target'] = 0
    #
    #         df = pd.concat([true_df, fake_df], ignore_index=True, sort=False)
    #         df = df.shuffle()
    #         df = self.del_unused(df)
    #         df.head()
    #         pickle.dump(df, open(filename, 'wb'))
    #         return df

 # @staticmethod
 #    def del_unused(df):
 #        df['text'] = df['title'] + ' ' + df['text']
 #        del df['title']
 #        del df['subject']
 #        del df['date']
 #        return df

# @staticmethod
# def get_xgb_model(X_train, y_train):  # 90
#     filename = 'xgb_model.sav'
#     path = Path(filename)
#     if path.is_file():
#         model = pickle.load(open(filename, 'rb'))
#         return model
#     else:
#         param_range = [1, 2, 3, 4, 5, 6]
#         param_range_fl = [1.0, 0.5, 0.1]
#         n_estimators = [50, 100, 150]
#         learning_rates = [.1, .2, .3]
#         xgb_param_grid = [{'XGB__learning_rate': learning_rates,
#                            'XGB__max_depth': param_range,
#                            'XGB__min_child_weight': param_range[:2],
#                            'XGB__subsample': param_range_fl,
#                            'XGB__n_estimators': n_estimators}]
#         model = GridSearchCV(XGBClassifier(random_state=42),
#                              param_grid=xgb_param_grid,
#                              scoring='accuracy',
#                              cv=3)
#         model.fit(X_train, y_train)
#         pickle.dump(model, open(filename, 'wb'))
#         return model