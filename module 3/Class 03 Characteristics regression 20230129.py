# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 4)

# Define a path for import and export
path = './'

# Import data
returns1 = pd.read_excel(path + 'Excel03 Data 20230128.xlsx', sheet_name='ret06')

print(returns1.head())
print(returns1.columns)

# Regression using statsmodels
# Adjusted abnormal stock returns regressed on LN mkt cap and industry indicator variables
# We will not add a constant because we have an industry indicator variable for every observation
# I have left in the add constant code, but it is swtiched off
y = returns1['abretadj']
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
              'd_4900_Communications',
              'd_6000_Miscellaneous']]
# x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
# To generate clustered standard errors use the line below
# model = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': returns1['CUSIP']})
predictions = model.predict(x)
print_model = model.summary()
b_coef = model.params
b_err = model.bse

influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
analysis = pd.DataFrame({'predictions': predictions, 'cooks_d': cooks_d})

print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(b_coef)
print(b_err)
print(analysis)

# Regression using statsmodels
# Unadjusted abnormal stock returns regressed on LN mkt cap and industry indicator variables
# We will not add a constant because we have an industry indicator variable for every observation
# I have left in the add constant code, but it is swtiched off
y1 = returns1['abret']
x1 = returns1[['lag1lnmc',
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
               'd_4900_Communications',
               'd_6000_Miscellaneous']]
# x = sm.add_constant(x)
model1 = sm.OLS(y1, x1).fit()
# To generate clustered standard errors use the line below
# model = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': returns1['CUSIP']})
predictions1 = model1.predict(x1)
print_model1 = model1.summary()
b_coef1 = model1.params
b_err1 = model1.bse

influence1 = model1.get_influence()
cooks_d1 = influence1.cooks_distance[0]
analysis1 = pd.DataFrame({'predictions': predictions1, 'cooks_d': cooks_d1})

print(print_model1)

print(f'R-squared: {model1.rsquared:.4f}')
print(b_coef1)
print(b_err1)
print(analysis1)

# Regression using statsmodels
# Adjusted abnormal stock returns regressed on industry indicator variables
# We will not add a constant because we have an industry indicator variable for every observation
# I have left in the add constant code, but it is swtiched off
y2 = returns1['abretadj']
x2 = returns1[[
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
    'd_4900_Communications',
    'd_6000_Miscellaneous']]
# x = sm.add_constant(x)
model2 = sm.OLS(y2, x2).fit()
# To generate clustered standard errors use the line below
# model = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': returns1['CUSIP']})
predictions2 = model2.predict(x2)
print_model2 = model2.summary()
b_coef2 = model2.params
b_err2 = model2.bse

influence2 = model2.get_influence()
cooks_d2 = influence2.cooks_distance[0]
analysis2 = pd.DataFrame({'predictions': predictions2, 'cooks_d': cooks_d2})

print(print_model2)
print(f'R-squared: {model2.rsquared:.4f}')
print(b_coef2)
print(b_err2)
print(analysis2)
