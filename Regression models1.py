#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("Downloads/Algerian_forest_fires_dataset_UPDATE.csv", skiprows=1)


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df[df.duplicated()]


# In[8]:


df.day.str.isnumeric().sum()


# In[9]:


df[~df['day'].str.isnumeric()]


# In[10]:


df=df.drop(df.index[[122,123]])


# In[11]:


df.shape


# In[68]:


df["day"]=df["day"].astype(int)


# In[13]:


df["day"].dtype


# In[14]:


df["month"].str.isnumeric().sum()


# In[69]:


df["month"]=df["month"].astype(float)


# In[16]:


df["month"].dtype


# In[70]:


df["year"].str.isnumeric().sum()


# In[71]:


df["year"]=df["year"].astype(float)


# In[19]:


df["Temperature"].str.isnumeric().sum()


# In[13]:


df["Temperature"]=df["Temperature"].astype(float)


# In[16]:


df[df["FFMC"].str.isnumeric()]


# In[15]:


df[~df[''].str.isnumeric()]


# In[17]:


df["FFMC"]=df["FFMC"].astype(float)


# In[65]:


df["DMC"].str.isnumeric().sum()


# In[66]:


df[~df["DMC"].str.isnumeric()]


# In[67]:





# In[18]:


df["DC"].str.isnumeric().sum()


# In[23]:


df[~df["DC"].str.isnumeric()]


# In[25]:


df["DC"]=df["DC"].replace('14.6 9','14.69')


# In[26]:


df["DC"]=df["DC"].astype(float)


# In[27]:


df["ISI"].str.isnumeric().sum()


# In[28]:


df[~df["ISI"].str.isnumeric()]


# In[29]:


df["ISI"]=df["ISI"].astype(float)


# In[30]:


df["BUI"].str.isnumeric().sum()


# In[31]:


df[~df["BUI"].str.isnumeric()]


# In[32]:


df["BUI"]=df["BUI"].astype(float)


# In[34]:


df["FWI"].str.isnumeric().sum()


# In[35]:


df[~df["FWI"].str.isnumeric()]


# In[40]:


df["FWI"]=df["FWI"].replace('fire   ', ' 0')


# In[41]:


df["FWI"]=df["FWI"].astype(float)


# In[56]:


df[" RH"].str.isnumeric().sum()


# In[57]:


df[" RH"]=df[" RH"].astype(float)


# In[58]:


df[" Ws"].str.isnumeric().sum()


# In[59]:


df[" Ws"]=df[" Ws"].astype(float)


# In[60]:


df["Rain "].str.isnumeric().sum()


# In[62]:


df[~df["Rain "].str.isnumeric()]


# In[63]:


df["Rain "]=df["Rain "].astype(float)


# In[72]:


df.info()


# In[78]:


df.columns


# In[73]:


df.isnull().sum()


# In[74]:


df.corr()


# In[75]:


sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(df.corr(), annot=True)


# In[81]:


df.drop(columns=["Classes  "], inplace=True)


# In[82]:


df


# In[83]:


## Divide the independent and dependent feature
X=df.drop(columns=["Temperature"])
y=df["Temperature"]


# In[84]:


X


# In[85]:


y


# In[86]:


## split the train and test data
from sklearn.model_selection import train_test_split


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[88]:


X_train


# In[89]:


X_train.shape


# In[90]:


X_test.shape


# # feature engineering

# In[92]:


## standardization
from sklearn.preprocessing import StandardScaler   ### mean=0, std=1


# In[93]:


scaler=StandardScaler() ## created an object


# In[94]:


scaler


# In[95]:


X_train=scaler.fit_transform(X_train)


# In[96]:


X_test=scaler.transform(X_test) ### fit is not used to avoid data leakage, model should not learn X_train data


# In[97]:


X_train


# In[98]:


X_test


# In[99]:


## model training
from sklearn.linear_model import LinearRegression


# In[100]:


reg = LinearRegression()


# In[101]:


reg


# In[102]:


reg.fit(X_train, y_train)


# In[103]:


## print the coefficient and intercept
print(reg.coef_)


# In[104]:


reg.intercept_


# In[105]:


##prediction
reg_predict=reg.predict(X_test)


# In[106]:


reg_predict


# # Assumptions of linear regression

# In[108]:


plt.scatter(y_test, reg_predict) ### showing linear relationship, model is good
plt.xlabel("Test truth data")
plt.ylabel("test predicted data") ## here x is inc when y is inc


# In[109]:


## residual
residuals=y_test-reg_predict


# In[110]:


residuals


# In[111]:


sns.displot(residuals, kind="kde")  ## should follow a normal distribution, but this is slighly right skewed,
#outliers are present on right


# In[112]:


## scatter plot with predictions and residual
## uniform distribution
plt.scatter(reg_predict, residuals) ## randomly distribution, uniform distribution (no shape)
##model is good


# In[113]:


## performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test, reg_predict))
print(mean_absolute_error(y_test, reg_predict))
print(np.sqrt(mean_squared_error(y_test, reg_predict)))


# In[114]:


## r squared and adj r squared   
## adj r will always be less than r squared
from sklearn.metrics import r2_score
score=r2_score(y_test, reg_predict)
score


# In[115]:


## adj r squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# # Ridge Regression 

# In[117]:


## ridge regression 
from sklearn.linear_model import Ridge
ridge=Ridge()


# In[118]:


ridge.fit(X_train, y_train)


# In[121]:


ridge_predict=ridge.predict(X_test)
ridge_predict


# In[122]:


##Assumptions of linear regression
plt.scatter(y_test, ridge_predict) ### showing linear relationship, model is good
plt.xlabel("Test truth data")
plt.ylabel("test predicted data") ## here x is inc when y is inc


# In[124]:


## residual
residuals=y_test-ridge_predict
residuals


# In[125]:


sns.displot(residuals, kind="kde")  ## should follow a normal distribution, but this is slighly right skewed,
#outliers are present on right


# In[128]:


## scatter plot with predictions and residual
## uniform distribution
plt.scatter(ridge_predict, residuals) ## randomly distribution, uniform distribution (no shape)
##model is good


# In[129]:


## performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test, ridge_predict))
print(mean_absolute_error(y_test, ridge_predict))
print(np.sqrt(mean_squared_error(y_test, ridge_predict)))


# In[130]:


## r squared and adj r squared   
## adj r will always be less than r squared
from sklearn.metrics import r2_score
score=r2_score(y_test, ridge_predict)
score


# In[131]:


## adj r squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# In[132]:


## Lasso Regression 
from sklearn.linear_model import Lasso


# In[133]:


lasso=Lasso()


# In[134]:


lasso


# In[135]:


lasso.fit(X_train, y_train)


# In[136]:


## print the coefficient and intercept
print(lasso.coef_)


# In[137]:


lasso.intercept_


# In[138]:


##prediction
lasso_predict=lasso.predict(X_test)


# In[139]:


lasso_predict


# In[141]:


plt.scatter(y_test, lasso_predict) ### showing linear relationship, model is good
plt.xlabel("Test truth data")
plt.ylabel("test predicted data") ## here x is inc when y is inc


# In[142]:


## residual
residuals=y_test-lasso_predict
residuals


# In[143]:


sns.displot(residuals, kind="kde")  ## should follow a normal distribution, but this is slighly right skewed,
#outliers are present on right


# In[144]:


## scatter plot with predictions and residual
## uniform distribution
plt.scatter(lasso_predict, residuals) ## randomly distribution, uniform distribution (no shape)
##model is good


# In[145]:


## performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test, lasso_predict))
print(mean_absolute_error(y_test, lasso_predict))
print(np.sqrt(mean_squared_error(y_test, lasso_predict)))


# In[146]:


## r squared and adj r squared   
## adj r will always be less than r squared
from sklearn.metrics import r2_score
score=r2_score(y_test, lasso_predict)
score


# In[147]:


## adj r squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# # ElasticNet Regression

# In[148]:


from sklearn.linear_model import ElasticNet


# In[149]:


elastic=ElasticNet()


# In[150]:


elastic


# In[151]:


elastic.fit(X_train, y_train)


# In[152]:


## print the coefficient and intercept
print(elastic.coef_)


# In[153]:


elastic.intercept_


# In[155]:


##prediction
elastic_predict=elastic.predict(X_test)


# In[156]:


elastic_predict


# In[157]:


plt.scatter(y_test, elastic_predict) ### showing linear relationship, model is good
plt.xlabel("Test truth data")
plt.ylabel("test predicted data") ## here x is inc when y is inc


# In[158]:


## residual
residuals=y_test-elastic_predict
residuals


# In[159]:


sns.displot(residuals, kind="kde")  ## should follow a normal distribution, but this is slighly right skewed,
#outliers are present on right


# In[160]:


## scatter plot with predictions and residual
## uniform distribution
plt.scatter(elastic_predict, residuals) ## randomly distribution, uniform distribution (no shape)
##model is good


# In[161]:


## performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test, elastic_predict))
print(mean_absolute_error(y_test, elastic_predict))
print(np.sqrt(mean_squared_error(y_test, elastic_predict)))


# In[162]:


## r squared and adj r squared   
## adj r will always be less than r squared
from sklearn.metrics import r2_score
score=r2_score(y_test, elastic_predict)
score


# In[163]:


## adj r squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# # conclusion
# ## Out of all the above four regression models, linear Regression has performed well and has learnt the model and predicted the temperature well.
# 

# In[ ]:




