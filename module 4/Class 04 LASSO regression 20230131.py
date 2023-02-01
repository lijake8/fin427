#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[44]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import math
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline


# Change the number of rows and columns to display

# In[119]:


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 4)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# Define a path for import and export

# In[6]:


path = 'E:\\UM\Courses\\FIN427\\FIN427 Winter 2023\\'


# Import data and create a quarter variable for alignment with quarterly financial information

# In[9]:


returns1 = pd.read_excel(path + 'Excel03 Data 20230128.xlsx', sheet_name='ret06')


# In[123]:


returns1['quarter'] = returns1['month'] + pd.offsets.QuarterEnd()
print(returns1.head())


# In[48]:


print(returns1.head())
print(returns1.columns)


# In[93]:


sales = pd.read_excel(path + 'SP400_Sales_20230131.xlsx', sheet_name='Sales')


# In[175]:


sales1 = sales
sales1 = sales1.dropna()
sales1 = sales1[sales1['Sales/Turnover (Net)'] > 0]
sales1['sales'] = sales1['Sales/Turnover (Net)']
sales1['quarter'] = sales1['Data Date'] + pd.offsets.QuarterEnd()
sales1['leadquarter'] = sales1['Data Date'] + pd.offsets.QuarterEnd(1)
sales1['lnsales'] = np.log(sales1['sales']*1000)
univ_sales = sales1.sales.describe()
univ_lnsales = sales1.lnsales.describe()
sales1['cusip8'] = sales1['CUSIP'].str[0:8]
print(univ_sales)
print(univ_lnsales)
print(sales1.head())
print(sales1.columns)
print(sales1)


# Run an OLS regression of abnormal returns on characteristics in which the variables are transformed to z-scores. Industry indicator variables are not included.

# In[12]:


y1 = returns1['abretadj']
x  = returns1[['lag1lnmc']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)


# Run an LASSO regression of abnormal returns on characteristics in which the variables are transformed to z-scores. Industry indicator variables are not included.

# In[30]:


y1 = returns1['abretadj']
x  = returns1[['lag1lnmc']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
lasso = Lasso(alpha = 0.001)
lasso.fit(x1, y1)
lasso_coef = lasso.fit(x1, y1).coef_
lasso_score = lasso.score(x1, y1)
print(f'Lasso score: {lasso_score: .4f}')
print(lasso_coef)
# print(pd.Series(lasso_coef, index = x.columns)) #This only works if the constant is excluded.


# Run an OLS regression of abnormal returns on characteristics in which the variables are transformed to z-scores. Industry indicator variables are included. There is an intercept and one industry (6000 Miscellaneous) is omitted. The intercept is the baseline monthly return on 6000 Miscellaneous before consideration of relative market capitalisation, and the coeffficients on other industries are incremental returns associated with those industries.

# In[16]:


y1 = returns1['abretadj']
x  = returns1[['lag1lnmc', 
    'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables', 
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services', 
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)
# print(pd.Series(ols_coef, index = x.columns))


# Run a LASSO regression of abnormal returns on characteristics in which the variables are transformed to z-scores. Industry indicator variables are included. There is an intercept and one industry (6000 Miscellaneous) is omitted. The intercept is the baseline monthly return on 6000 Miscellaneous before consideration of relative market capitalisation, and the coeffficients on other industries are incremental returns associated with those industries.

# In[27]:


y1 = returns1['abretadj']
x = returns1[['lag1lnmc', 
    'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables', 
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services', 
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
lasso = Lasso(alpha = 0.001)
lasso.fit(x1, y1)
lasso_coef = lasso.fit(x1, y1).coef_
lasso_score = lasso.score(x1, y1)
print(f'Lasso score: {lasso_score: .4f}')
print(lasso_coef)
print(x.columns)
# print(pd.Series(lasso_coef, index = x.columns)) #This only works if the constant is excluded.


# Run an OLS regression of abnormal returns on characteristics in which the variables are transformed to z-scores, and do the same for untransformed variables. Including only variables identified as material by the LASSO regression. Industry indicator variables are not included.

# In[35]:


y1 = returns1['abretadj']
x  = returns1[['lag1lnmc']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)

x = sm.add_constant(x)
model2 = sm.OLS(y1, x).fit()
print_model2 = model2.summary()
ols_coef2 = model2.params
ols_rsq2  = model2.rsquared
print(print_model2)
print(f'R-squared: {model2.rsquared:.4f}')
print(ols_coef2)


# Run an OLS regression of abnormal returns on characteristics in which the variables are transformed to z-scores, and do the same for untransformed variables. Including only variables identified as material by the LASSO regression. Industry indicator variables are included.

# In[34]:


y1 = returns1['abretadj']
x  = returns1[['lag1lnmc', 'd_1300_Electronic_Technology', 'd_2300_Health_Technology', 'd_3300_Technology_Services']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)

x = sm.add_constant(x)
model2 = sm.OLS(y1, x).fit()
print_model2 = model2.summary()
ols_coef2 = model2.params
ols_rsq2  = model2.rsquared
print(print_model2)
print(f'R-squared: {model2.rsquared:.4f}')
print(ols_coef2)


# Merge returns1 and sales1 dataframes

# In[176]:


combined1 = returns1.merge(sales1, how='inner', left_on=['CUSIP', 'quarter'], right_on=['cusip8', 'leadquarter'])
combined1.rename(columns={'sales': 'lag1sales', 'lnsales': 'lag1lnsales'}, inplace=True)
print(combined1.head())
print(combined1.columns)
print(len(returns1.index))
print(len(sales1.index))
print(len(combined1.index))
print(combined1.lag1sales.describe(percentiles=[0.125, 0.875]))
print(combined1.lag1mc.describe(percentiles=[0.125, 0.875]))
print(combined1.lag1lnsales.describe(percentiles=[0.125, 0.875]))
print(combined1.lag1lnmc.describe(percentiles=[0.125, 0.875]))
print(combined1.retadj.describe(percentiles=[0.125, 0.875]))
print(combined1.bmret.describe(percentiles=[0.125, 0.875]))
print(combined1.abretadj.describe(percentiles=[0.125, 0.875]))


# Compute the correlation between lnsales and lag1lnmc

# In[114]:


dfcorr = combined1[['lag1lnsales', 'lag1lnmc']].copy()
dfcorr.corr(method='pearson')


# Run an OLS regression with two size variables, lag1lnmc and lag1lnsales. Run the regression again, with just lag1mc or lag1sales, and compare the results. Show how the positive correlation between market cap and sales leads to one coefficient being positive and another coefficient being negative. Repeat using z-scores.

# In[190]:


y1 = combined1['abretadj']
# x  = combined1[['lag1lnmc']]
# x  = combined1[['lag1lnsales']]
x  = combined1[['lag1lnmc', 'lag1lnsales']]
x1 = StandardScaler().fit_transform(x)
x = sm.add_constant(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)


# Now incorporate industry indicator variables, and run the regressions again with two size variables.

# In[191]:


y1 = combined1['abretadj']
x  = combined1[['lag1lnmc', 'lag1lnsales', 
    'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables', 
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services', 
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
# x  = combined1[['lag1lnmc']]
# x  = combined1[['lag1lnsales']]
x1 = StandardScaler().fit_transform(x)
x = sm.add_constant(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)


# On the combined dataset, with two measures of size, run a LASSO regression of abnormal returns on characteristics in which the variables are transformed to z-scores. Industry indicator variables are excluded. There is an intercept and one industry (6000 Miscellaneous) is omitted. The intercept is the baseline monthly return on 6000 Miscellaneous before consideration of relative market capitalisation, and the coeffficients on other industries are incremental returns associated with those industries.

# In[193]:


y1 = combined1['abretadj']
x = combined1[['lag1lnmc']]
# x = combined1[['lag1lnmc', 'lag1lnsales']]
# x = combined1[['lag1lnmc', 'lag1lnsales']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
lasso = Lasso(alpha = 0.001)
lasso.fit(x1, y1)
lasso_coef = lasso.fit(x1, y1).coef_
lasso_score = lasso.score(x1, y1)
print(f'Lasso score: {lasso_score: .4f}')
print(lasso_coef)
print(x.columns)
# print(pd.Series(lasso_coef, index = x.columns)) #This only works if the constant is excluded.


# In[209]:


y1 = combined1['abretadj']
x = combined1[['lag1lnmc', 'lag1lnsales',
    'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables', 
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services', 
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
lasso = Lasso(alpha = 0.0014)
lasso.fit(x1, y1)
lasso_coef = lasso.fit(x1, y1).coef_
lasso_score = lasso.score(x1, y1)
print(f'Lasso score: {lasso_score: .4f}')
print(lasso_coef)
print(x.columns)
# print(pd.Series(lasso_coef, index = x.columns)) #This only works if the constant is excluded.


# In[196]:


y1 = combined1['abretadj']
x = combined1[['lag1lnmc', 'lag1lnsales']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
lasso = Lasso(alpha = 0.001)
lasso.fit(x1, y1)
lasso_coef = lasso.fit(x1, y1).coef_
lasso_score = lasso.score(x1, y1)
print(f'Lasso score: {lasso_score: .4f}')
print(lasso_coef)
print(x.columns)
# print(pd.Series(lasso_coef, index = x.columns)) #This only works if the constant is excluded.


# OLS regression incorporating only those industries identified by LASSO as being material.

# In[201]:


y1 = combined1['abretadj']
x = combined1[['lag1lnmc','lag1lnsales',
   'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables', 
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services', 
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
x1 = StandardScaler().fit_transform(x)
x = sm.add_constant(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)


# In[197]:


y1 = combined1['abretadj']
x = combined1[['lag1lnmc','lag1lnsales']]
x1 = StandardScaler().fit_transform(x)
x = sm.add_constant(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)

