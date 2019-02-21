#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


Admissions = pd.read_csv('Admission_Predict.csv')


# In[3]:


Admissions.head()


# In[4]:


Admissions.info()


# In[5]:


Admissions.describe()


# In[6]:


Admissions.columns


# In[7]:


import seaborn as sns


# In[8]:


sns.pairplot(Admissions)


# In[9]:


sns.distplot(Admissions['GRE Score'])


# In[10]:


Admissions.corr()


# In[ ]:





# In[13]:


type(Admissions)


# In[91]:


X = np.array(Admissions['CGPA'])


# In[92]:


Y = np.array(Admissions['GRE Score'])


# In[93]:


from sklearn.linear_model import LinearRegression


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X = X.reshape(1, -1)
Y = Y.reshape(1, -1)


# In[96]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)


# In[97]:


lm = LinearRegression()


# In[98]:


lm.fit(X_test, Y_test)


# In[99]:


predictions = lm.predict(X_test)


# In[100]:


plt.scatter(Y_test,predictions)


# In[ ]:





# In[ ]:





# In[ ]:




