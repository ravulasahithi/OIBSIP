#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv("car data.csv")


# In[3]:


df.head()


# In[7]:


df=df.drop(columns=["Car_Name"])


# In[8]:


df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df["Selling_Price"].describe()


# In[11]:


df["Selling_Price"].plot.hist()


# In[12]:


sns.boxplot(df["Selling_Price"])


# In[15]:


q1=df["Selling_Price"].quantile(0.25)
q3=df["Selling_Price"].quantile(0.75)
IQR=q3-q1
IQR


# In[16]:


upper_limit = q3+1.5*IQR
lower_limit = q1-1.5*IQR
upper_limit,lower_limit


# In[17]:


def imputer(value):
    if value>upper_limit:
        return upper_limit
    if value<lower_limit:
        return lower_limit
    else:
        return value


# In[18]:


df["Selling_Price"]=df["Selling_Price"].apply(imputer)


# In[19]:


df.info()


# In[20]:


x=df.drop(columns=["Selling_Price"])
y=df["Selling_Price"]


# In[21]:


categorical=x.select_dtypes(include=["object"])


# In[22]:


dummy = pd.get_dummies(categorical,drop_first=True)
x= x.drop(list(categorical.columns),axis=1)
x=pd.concat([x,dummy],axis=1)


# In[24]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test, y_train,y_test=tts(x, y,random_state=101)


# In[27]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor() 
model.fit(x_train, y_train)
predictions= model.predict(x_test)


# In[28]:


from sklearn.metrics import mean_absolute_error
model.score(x_test, predictions)


# In[ ]:




