#!/usr/bin/env python
# coding: utf-8

# # Project - Perceptron
# - Try a Perceptron model with more dimensions

# ### Step 1: Import libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 2: Read the data
# - Use Pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method to read **files/weather.csv**
# - HINT: Use **parse_dates=True** and **index_col=0**

# In[2]:


data = pd.read_csv('files/weather.csv', parse_dates=True, index_col=0)


# In[3]:


data.head()


# ### Step 3: Investigate data
# - Look for missing data points
# - You can do that by applying **isna()** and **sum()**, which will give a summary of rows missing data for each column.
#     - Resource: [isna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html)
#     - Resource: [sum()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html)

# In[4]:


data.isna().sum()


# In[5]:


data.isnull().sum()


# ### Step 4: Remove 'dirty' columns 
# - Make a choice and remove columns with too many entries with NaN.
# - Say, take all columns with more than 100 rows.
# - Also, you can remove rows with non-numeric values (remember to keep **RainTomorrow**)
# - To remove rows use [drop(columns, axis=1)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)

# In[6]:


dataset = data.drop(['WindGustDir', 'WindGustSpeed', 'Cloud9am', 'Cloud3pm', 'WindDir9am', 'WindDir3pm', 'RainToday'], axis=1)


# In[7]:


dataset.head()


# ### Step 5: Deal with remaining missing data
# - A simple choice is to simply remove rows with missing data
# - Use [dropna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)

# In[8]:


dataset_clean = dataset.dropna()


# In[9]:


len(dataset), len(dataset_clean)


# ### Step 6: Create training and test datasets
# - Define dataset **X** to consist of all data except **'RainTomorrow'**.
#     - Use [dropna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
# - Define dataset **y** to be datset cosisting of **'RainTomorrow'**.
# - Divide into **X_train, X_test, y_train, y_test** with **train_test_split**
#     - You can use **random_state=42** (or any other number) if you want to reproduce results.

# In[10]:


X = dataset_clean[dataset_clean.columns[:-1]]
y = dataset_clean['RainTomorrow']
y = np.array([0 if value == 'No' else 1 for value in y])


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Step 7: Train and test the model
# - Create classifier with **Perceptron**
#     - You can use **random_state=0** to be able to reproduce
# - Fit the model with training data **(X_train, y_train**)
# - Predict data from **X_test** (use predict) and assign to **y_pred**.
# - Evalute score by using **metrics.accuracy_score(y_test, y_pred)**.
# - You can redo with different choice of columns

# In[12]:


clf = Perceptron(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[13]:


sum(y == 0) / len(y)


# ### Step 8 (Optional): Plot the result
# - Use Matplotlib.pyplot (**plt**) with **subplots** to create a figure and axes (**fig, ax**)
# - Predict all the datapoints in **X**.
# - Make a scatter plot with all datapoints in **X** with color by the predictions made.
#     - You might want to use **alpha=0.25** in your plot as argument.

# In[14]:


fig, ax = plt.subplots()
y_pred = clf.predict(X)
ax.scatter(x=X['Humidity3pm'], y=X['Pressure3pm'], c=y_pred, alpha=.25)


# In[ ]:





# In[ ]:




