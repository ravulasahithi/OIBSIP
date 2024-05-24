#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[2]:


ds= pd.read_csv("spam.csv",encoding="ISO-8859-1")
ds


# In[3]:


ds = ds[['v1','v2']]
ds.columns = ['category','message']
ds


# In[4]:


ds.groupby('category').describe()


# In[5]:


encoder = LabelEncoder()
ds['category'] = encoder.fit_transform(ds['category'])
ds


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(ds.message,ds.category,test_size=0.2)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()

x_train_count = vec.fit_transform(x_train.values)

x_train_count.toarray()


# In[9]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_count,y_train)


# In[10]:


x_test_count = vec.transform(x_test)
accuracy = model.score(x_test_count,y_test)
print(f'Accuracy of model:{accuracy*100:.2f}%')


# In[ ]:


def predict():
    message = input("Enter the message to predict")
    message = [message]
    vector = vec.transform(message)
    if(model.predict(vector)==1):
        print("SPAM EMAIL")
    else:
        print("NOT SPAM")


# In[ ]:


predict()


# In[ ]:





# In[ ]:




