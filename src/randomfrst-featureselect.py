
# coding: utf-8

# In[2]:


import numpy as np
from numpy import *
from  sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import RandomForestClassifier

import sklearn.grid_search
from sklearn import linear_model, decomposition
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA


# In[3]:


train_data = loadtxt('data/train.dat')


# In[5]:


train_labels = loadtxt('data/train.labels')


# In[6]:


test_data = loadtxt('data/test.dat')


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_labels, test_size = 0.2)


# In[8]:


select = sklearn.feature_selection.SelectKBest(k=100)


# In[10]:


forest = RandomForestClassifier()


# In[11]:


pipe = Pipeline(steps=[('feature_selection', select), ('random_forest', forest)])


# In[12]:


pipe.fit(X_train, Y_train)
test_predictions = pipe.predict(X_test)
np.mean(test_predictions==Y_test)


# In[13]:


pipe.fit(train_data, train_labels)
predict = pipe.predict(train_data)
np.mean(predict==train_labels)


parameters = dict(feature_selection__k=[50, 100, 150, 200], 
              random_forest__n_estimators=[50, 100, 150, 200],
              random_forest__min_samples_split=[2, 3, 4, 5, 6, 7, 8, 9, 10])

cv = sklearn.grid_search.GridSearchCV(pipe, param_grid=parameters, verbose=2)


# In[15]:


cv.fit(X_train, Y_train)
predict = cv.predict(X_test)


# In[16]:


np.mean(predict==Y_test)


# In[17]:


cv.best_params_


# In[18]:


pipe.set_params(**cv.best_params_);


# In[19]:


pipe.fit(X_train, Y_train)
predict = cv.predict(X_test)
np.mean(predict==Y_test)


# In[20]:


pipe.fit(train_data, train_labels)
predict = cv.predict(train_data)
np.mean(predict==train_labels)


# In[26]:


test_predict = pipe.predict(test_data)


# In[27]:


with open("/home/pavana/PR2/submission11/output.dat","w+") as f:
    for i in test_predict:
        f.write("%d\n"%(i))

