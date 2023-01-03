#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd

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

# In[34]:


import matplotlib.pyplot as plt
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()


# In[58]:


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns

# plt.figure(figsize=(15, 5))
# ax = plt.subplot(1, 2, 1)
# sns.countplot('subject', data=fake_df)
# plt.xticks(rotation=90)
# plt.ylabel('News count')
# plt.xlabel('Subject (News categories)')
# plt.title('Fake news dataset')
#
#
# plt.subplot(1, 2, 2)
# sns.countplot('subject', data=true_df)
# plt.ylabel('News count')
# plt.xlabel('Subject (News categories)')
# plt.title('True news dataset')
# plt.show()


# In[72]:


import re
import string

df['text'] = df['title'] + ' ' + df['text']  # 'politicsNews About trump he was a president'

del df['title']
del df['subject']
del df['date']

samp = df.sample(1)
text = samp.iloc[0]['text']
print(text)
text = re.sub(r'\d+', '', text)
print('Sentence After removing numbers\n', text)

text = text.translate(text.maketrans("", "", string.punctuation))
print('Sentence After Removing Punctuations\n', text)
text = re.sub('[^a-zA-Z]', ' ', text)  # replaces non-alphabets with spaces
text = text.lower()  # Converting
# In[73]:


import nltk

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
words_in_text = list(set(text.split(' ')) - stop_words)
print(words_in_text)

# In[76]:


from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')
pstemmer = PorterStemmer()
for i, word in enumerate(words_in_text):
    words_in_text[i] = pstemmer.stem(word)
print(words_in_text)

# Lemmatization of Words
# Lemmatisation is the process of grouping together the different inflected forms of a word so they can be analysed as a single item. Ex: dogs -> dog. I am not clear with difference between lemmatization and stemming. In most of the tutorials, I found them both and I could not understand the clear difference between the two.

from nltk.stem import WordNetLemmatizer

nlemmatizer = WordNetLemmatizer()
words = []
for i, word in enumerate(words_in_text):
    words_in_text[i] = nlemmatizer.lemmatize(word)
print(words_in_text)

# In[77]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

# In[78]:


word_count = {}
word_count_true = {}
word_count_false = {}
true = 0
false = 0

import re
import string
import nltk

stop_words = set(nltk.corpus.stopwords.words('english'))
from nltk.stem import PorterStemmer

pstemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer

wnlemmatizer = WordNetLemmatizer()

# In[82]:


row_count = train.shape[0]
for row in range(0, row_count):
    true += train.iloc[row]['target']
    false += (1 - train.iloc[row]['target'])
    text = train.iloc[row]['text']
    text = re.sub(r'\d+', '', text)
    text = text.translate(text.maketrans("", "", string.punctuation))
    words_in_text = list(set(text.split(' ')) - stop_words)
    for index, word in enumerate(words_in_text):
        word = pstemmer.stem(word)
        words_in_text[index] = wnlemmatizer.lemmatize(word)
    for word in words_in_text:
        if train.iloc[row]['target'] == 0:  # fake
            if word in word_count_false.keys():
                word_count_false[word] += 1
            else:
                word_count_false[word] = 1
        elif train.iloc[row]['target'] == 1:  # truth
            if word in word_count_true.keys():
                word_count_true[word] += 1
            else:
                word_count_true[word] = 1
        if word in word_count.keys():  # For all words. I use this to compute probability.
            word_count[word] += 1
        else:
            word_count[word] = 1

print('Done')

# In[ ]:


word_probability = {}
total_words = 0
for i in word_count:
    total_words += word_count[i]
for i in word_count:
    word_probability[i] = word_count[i] / total_words

# Removing words with occurance probability of 0.0001.
print('Total words ', len(word_probability))
print('Minimum probability ', min(word_probability.values()))
threshold_p = 0.0001
for i in list(word_probability):
    if word_probability[i] < threshold_p:
        del word_probability[i]
        if i in list(word_count_false):  # list(dict) return it;s key elements
            del word_count_false[i]
        if i in list(word_count_true):
            del word_count_true[i]
print('Total words ', len(word_probability))

# In[ ]:


total_fake_words = sum(word_count_false.values())
cp_false = {}  # Conditional Probability
for i in list(word_count_false):
    cp_false[i] = word_count_false[i] / total_fake_words

total_true_words = sum(word_count_true.values())
cp_true = {}  # Conditional Probability
for i in list(word_count_true):
    cp_true[i] = word_count_true[i] / total_true_words

# In[ ]:


row_count = test.shape[0]

p_true = true / (false + true)
p_false = false / (false + true)
accuracy = 0

for row in range(0, row_count):
    text = test.iloc[row]['text']
    target = test.iloc[row]['target']
    text = re.sub(r'\d+', '', text)
    text = text.translate(text.maketrans("", "", string.punctuation))
    words_in_text = list(set(text.split(' ')) - stop_words)
    for index, word in enumerate(words_in_text):
        word = pstemmer.stem(word)
        words_in_text[index] = wnlemmatizer.lemmatize(word)
    true_term = p_true
    false_term = p_false

    false_M = len(cp_false.keys())
    true_M = len(cp_true.keys())
    for word in words_in_text:
        if word not in cp_true.keys():
            true_M += 1
        if word not in cp_false.keys():
            false_M += 1

    for word in words_in_text:
        if word in cp_true.keys():
            true_term *= (cp_true[word] + (1 / true_M))
        else:
            true_term *= (1 / true_M)
        if word in cp_false.keys():
            false_term *= (cp_false[word] + (1 / false_M))
        else:
            false_term *= (1 / false_M)

    if true_term / (true_term + false_term) > 0.5:
        response = 1
    else:
        response = 0
    if target == response:
        accuracy += 1

print('Accuracy is ', accuracy / row_count * 100)

# In[ ]:
