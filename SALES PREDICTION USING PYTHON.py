#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[3]:


df = pd.read_csv('Advertising.csv')
df


# In[4]:


x = df.iloc[:,1:4]
x


# In[5]:


y=df.iloc[:,4]
y


# In[6]:


plt.scatter(df['TV'],y)
plt.title('TV Advertisements')


# In[7]:


plt.scatter(df['Radio'],y)
plt.title('Radio Advertisements')


# In[8]:


plt.scatter(df['Newspaper'],y)
plt.title('Newspaper Advertisements')


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()
cols = x.columns
x = scaler.fit_transform(x)
x=pd.DataFrame(x)
x.columns = cols
x


# In[12]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.25)


# In[17]:


from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(x_train,y_train)


# In[18]:


y_pre=clf.predict(x_test)
y_pre


# In[20]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pre)
print("mean squared Error",mse)


# In[24]:


import statsmodels.api as sm

x2=sm.add_constant(x)
est=sm.OLS(y,x2)
est2=est.fit()
print(est2.summary())


# In[ ]:




