#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


df=pd.read_csv('Iris.csv')


# In[3]:


df


# In[4]:


df.describe()


# In[7]:


data = df.values
X = data[:,1:4]
Y = data[:,5]
Y


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, Y,test_size=0.2)
y_train


# In[9]:


from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train,y_train)


# In[11]:


predictions = svn.predict(X_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,predictions)*100
print('%.2f'% acc)


# In[12]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:




