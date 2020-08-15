#!/usr/bin/env python
# coding: utf-8

# # REAL PRICING
# 
# #### The  Dataset is data about housing in the area of Boston.
# #### The goal of this machine learning project is to be able to make a price prediction of a house using linear regression and to determine the factors on which the price depends.
# 
# ##### The Steps involved are:
# #### 1. Understanding the data
# #### 2. Data Cleaning
# #### 3. Relationship Analysis
# #### 4. Selecting a model
# #### 5. Training and testing the model
# #### 6. Conclusion

#  ## 1.0 UNDERSTANDING THE DATA
# 
# ### 1.1 ImportIng required libraries:
# #### Since we are going to use various libraries for calculations, we need to import them.

# In[39]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1.2 Read the CSV file:
# #### The boston dataset of sklearn is imported. 
# #### We also pass the dataset to an object called "boston".  So going forward, we will refer to the dataset as "boston".

# In[40]:


from sklearn.datasets import load_boston
boston=load_boston()


# ### 1.3 Getting General Information about the data:
# 
# ####  We use the pandas Descr method to view some basic information about the data

# In[41]:


print(boston.DESCR)


# ### 1.4 Viewing the columns:
# #### Feature_names is used to view the column names in the data.

# In[42]:


boston.feature_names


# In[43]:


boston


# ### 1.5 The data is converted into a pandas dataframe (df). So going forward, we will be using 'df' to refer to the                                                                 dataset.

# In[44]:


df=pd.DataFrame(boston.data, columns=boston.feature_names)


# #### Now we want to view the first 5 rows of the dataframe.

# In[7]:


df.head()


# 
# 
# 
# 
# 

#  ## 2.0 CLEANING THE DATA
# 
# 
# #### The House prices column, which is going to be our target variable is missing in this dataframe.  So we need to add them to the dataframe and take a quick look at the data afterwards.

# In[46]:


df['MEDV']=boston.target


# In[47]:


df.head()


# #### To ensure quality of that data,we if there are null-values, or different types of values in one column. The info method will give us a quick overview about the data.
# 

# In[48]:


df.info()


# #### The results above shows that there are no null values or missing values in the data . So we can go ahead and perform a brief descrptive analysis of the data using the decsribe method.

# In[49]:


df.describe()


# 
# 
# 
# 
# 
# 

# ## 3.0 RELATIONSHIP ANALYSIS / VISUALISATION
# 
# #### Visualization will help us see the relationship between variables and to look at the price-distribution.
# 
# 
# ### First we will have a quick overview to see  the relationship between variables.

# In[50]:


sns.pairplot(df)


# ### The basic distribution of each column

# In[51]:


rows=7
cols=2

fig, ax=plt.subplots(nrows=rows, ncols=cols, figsize=(16,16))
col=df.columns
index=0

for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]], ax=ax[i][j])
        index=index+1
    
plt.tight_layout()


# ### Correlations 
# #### Next we check the correlations and summarize the relationships between the variables.
# 
# #### NB: a correlation factor near 1 or -1 is wanted.
# #### A correlation of 0 means that there is no linear relation between that columns and they will sabotage the linear regression model.

# In[52]:


fig, ax=plt.subplots(figsize=(16,9))
sns.heatmap(df.corr(), annot=True, annot_kws={'size':12})


# ## 4.0 SELECTING A MODEL
# 
# ### For this project we are using a linear regression model to predict the housing pricing (MEDV).
# 
# We need to select the best features in the data which can predict MEDV.
# 
# We do that by selecting features that have nearly high correlations with MEDV.
# In that case, we need to define a threshold filter(getCorrelatedFeature).

# In[53]:


def getCorrelatedFeature(corrdata, threshold):
    feature=[]
    value=[]
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshold:
            feature.append(index)
            value.append(corrdata[index])
    df=pd.DataFrame(data=value, index=feature, columns=['Corr Value'])
    
    return df


# #### Now we set the threshold to 0.4  and call the getCorrelatedFeature function to return a dataframe of all features which have a correlation of  more than 0.4 with MEDV.

# In[54]:


threshold=0.4
corr_value = getCorrelatedFeature(df.corr()['MEDV'], threshold)


# #### This is to display the name of features with correlation of more than 0.4. These are the features (Independent variables / predictors) we are going to use to predict MEDV(tartget variable/predictant).

# In[55]:


corr_value.index.values


# #### This is to pass the dataframe to an object called correlated_data and  have a quick view of the dataframe.     
# 
# The new dataframe, 'correlated_data' is the data that would be used for the analysis.

# In[56]:


correlated_data=df[corr_value.index]
correlated_data.head()


# 
# 
# 
# 
# 
# 

# ## 5.0 TRAINING AND TESTING THE MODEL
# 
# #### We will split the given data into a training and testing dataset , and fit the model to train and test data .

# In[57]:


X=correlated_data.drop(labels=['MEDV'], axis=1)
y=correlated_data['MEDV']


# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33,random_state=1)


# In[59]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()


# In[60]:


lm.fit(X_train,y_train)


# ### Result in a mathematical / visual way
# 
# #### First, we can display the results of the test in an array form.
# 
# 
# 
# 

# In[61]:


predictions=lm.predict(X_test)
predictions


# #### Then we can also visualize the test results in a scatterplot.
# 
# #### We want to have a perfect linear relation of the points (or nearly linear). The larger the distribution of points, the greater the inaccuracy of the model.

# In[62]:


plt.scatter(y_test,predictions)


# #### We then visualize in a histogram plot.

# In[63]:


sns.distplot((y_test-predictions),bins=50)


# #### Finding the intercept of the linear regression function

# In[64]:


lm.intercept_


# #### Finding the coefficients of the linear regression function

# In[65]:


lm.coef_


# #### Defining the linear regression function

# In[66]:


def lin_func(values, coefficients=lm.coef_, y_axis=lm.intercept_):
    return np.dot(values, coefficients)+y_axis


# 
# 
# #### This is to create a random test data to test the linear regression model.
# #### It displays the comparision between actual and predicted values of MEDV.

# In[67]:


from random import randint
for i in range(5):
    index=randint(0,len(df)-1)
    sample=df.iloc[index][corr_value.index.values].drop('MEDV')

    print(
        'PREDICTION:',round(lin_func(sample),2),
        '// REAL:',df.iloc[index]['MEDV'],
        '// DIFFERENCE:',round(round(lin_func(sample),2)-df.iloc[index]['MEDV'],2)
    )


# ## 6.0 CONCLUSION
# 
# #### The purpose of this project was to predict housing price and to determine the right predictors.
# #### It was seen from Section 4 that 'INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT' were the factors on which the price depends on, based on our correlation analysis in Section 3.
# #### The results of training the data also showed that there were differences between the actual prices and predicted prices. This is okay because errors are allowed in machine learning. However, errors are allowed only to some extent. 
# #### So the model can be evaluated to check for its accuracy in predicting housing prices.

# In[ ]:




