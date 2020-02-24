#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import itertools


# In[67]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv('data.csv')

for i in range(0,df.shape[0]-1):
    if(df.Body.isnull()[i]):
        df.Body[i] = df.Headline[i]


# In[68]:


labels=df.Label
labels.head()


# In[69]:


x_train,x_test,y_train,y_test=train_test_split(df['Body'], labels, test_size=0.2)


# In[70]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english')


# In[71]:


#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[72]:


clf = MultinomialNB() 
clf.fit(tfidf_train, y_train)                       # Fit Naive Bayes classifier according to X, y
pred = clf.predict(tfidf_test)                     # Perform classification on an array of test vectors X.
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(cm)

#Applying Passive Aggressive classifier
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(cm)


# In[ ]:




