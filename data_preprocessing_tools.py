#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing Tools

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[3]:


print(X)


# In[4]:


print(y)


# ## Taking care of missing data

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[ ]:


print(X)


# ## Encoding categorical data

# ### Encoding the Independent Variable

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[ ]:


print(X)


# ### Encoding the Dependent Variable

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[ ]:


print(y)


# ## Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[ ]:


print(X)


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


print(X_train)


# In[ ]:


print(X_test)


# In[ ]:


print(y_train)


# In[ ]:


print(y_test)

