#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #converting Text to␣numerical Values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


df= pd.read_csv("C:\\Users\\Ananya saha\\Downloads\\Projects\\Spam_Mail_Detection\\dataset\\mail_data.csv")


# In[5]:


df.head(5)


# In[6]:


df.shape


# In[7]:


df.where((pd.notnull(df)),'',inplace=True) #replace null values with null string


# In[8]:


df.isnull().sum() # to check null values 


# In[9]:


df.loc[df['Category']=='spam','Category',]=0
df.loc[df['Category']=='ham','Category',]=1


# In[10]:


#Separating features and target
X=df['Message']
Y=df['Category']


# In[11]:


print(X)


# In[12]:


print(Y)


# In[13]:


#Spliting data into Train data and Test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)


# In[14]:


print(X.shape,  X_train.shape,  X_test.shape)


# In[15]:


# Feature Extraction

#Converttextdatatonumericalvalues
vectorizer=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_feature=vectorizer.fit_transform(X_train)
X_test_feature=vectorizer.transform(X_test)

#ConvertY_train and Y_testvaluesasinteger
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# In[16]:


print(X_train_feature)


# In[17]:


print(X_test_feature)


# ## Model Training

# In[18]:


Model=LogisticRegression()


# In[19]:


Model.fit(X_train_feature,Y_train)


# ## Model Evaluation

# In[20]:


#Prediction on Training Data
prediction_on_training_data=Model.predict(X_train_feature)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy on training data :',accuracy_on_training_data)


# In[21]:


#Prediction on Test data
prediction_on_test_data=Model.predict(X_test_feature)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print('Accuracy on test data :',accuracy_on_test_data)

