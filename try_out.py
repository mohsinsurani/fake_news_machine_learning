#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# In[25]:


fake_df = pd.read_csv('archive/Fake.csv')
true_df = pd.read_csv('archive/True.csv')

# In[26]:


fake_df.head()

# In[27]:


true_df.head()

# In[28]:


true_df.isnull().values.any()
fake_df.isnull().values.any()

# In[29]:


true_df['target'] = 1
fake_df['target'] = 0

# In[30]:


df = pd.concat([true_df, fake_df], ignore_index=True, sort=False)

# In[31]:


df.head()

# In[32]:


totalLen = len(df['target'])
totalTrueVal = len(df[df['target'] == 1])
totalFalseVal = len(df[df['target'] == 0])
labels = 'True', 'False'
sizes = [totalTrueVal * 100 / totalLen, totalFalseVal * 100 / totalLen]

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

# Word_Cloud
from nltk.corpus import stopwords
from wordcloud import WordCloud

stop_words = stopwords.words('english')

comment_words = ''
stopwords = set(stop_words)

# iterate through the csv file
for val in df[df['target'] == 1]['text']:

    # typecaste each val to string
    val = str(val)

    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='black',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# import matplotlib.pyplot as plt
#
# # plot the WordCloud image
# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
#
# plt.show()

df['text'] = df['title'] + ' ' + df['text']  # 'politicsNews About trump he was a president'

del df['title']
del df['subject']
del df['date']

df.head()
import re

first_text = df['text'][0]
first_text

first_text = re.sub('\[[^]]*\]', ' ', first_text)  # remove punctuations
first_text = re.sub('[^a-zA-Z]', ' ', first_text)  # replaces non-alphabets with spaces
first_text = first_text.lower()  # Converting from uppercase to lowercase
first_text

import nltk
from nltk.corpus import stopwords

first_text = nltk.word_tokenize(first_text)
first_text = [word for word in first_text if not word in set(stopwords.words('english'))]

first_text = ' '.join(first_text)
first_text


def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)


# 알파벳이 아닌것 제거
def remove_characters(text):
    return re.sub('[^a-zA-Z]', ' ', text)


# 불용어 제거
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


def remove_stopwords(text):
    return ' '.join([word for word in nltk.word_tokenize(text) if word not in stop_words])


# str(text).split()
# 표제어 추출
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())


def clean_text(text):
    text = text.lower()
    text = remove_punctuations(text)
    text = remove_characters(text)
    text = remove_stopwords(text)
    text = lemmatize_words(text)
    return text


# apply
df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.3, random_state=0)

# tokenize
max_features = 10000  # 100 * 100
maxlen = 256  # avg maxlen 225



token = Tokenizer(num_words=max_features)
token.fit_on_texts(X_train)

# tokenize train
tokenized_train = token.texts_to_sequences(X_train)
x_train = pad_sequences(tokenized_train, maxlen=maxlen)

# tokenize test
tokenized_test = token.texts_to_sequences(X_test)
x_test = pad_sequences(tokenized_test, maxlen=maxlen)
