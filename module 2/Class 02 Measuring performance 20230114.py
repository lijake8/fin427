# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Define a path for import and export
path = './'

# Import monthly returns for individual S&P 400 stocks
df_stockret = pd.read_excel(path + 'Excel02 S&P 400 historical returns 20221231.xlsx', sheet_name='mth')
print(df_stockret.head())
print(df_stockret.columns)

# Import monthly returns on the S&P 400 index
df_indexret = pd.read_excel(path + 'Excel02 S&P 400 Returns 20221230.xlsx', sheet_name='Mth')
df_indexret['MthRet'] = df_indexret['RetInd'].pct_change()

print(df_indexret.head())
print(df_indexret.columns)

# Import monthly Fama and French factors
df_ff = pd.read_excel(path + 'Excel02 Fama and French 20230114.xlsx', sheet_name='Mth3')

print(df_ff.head())
print(df_ff.columns)

# Combine stock return data with index return data and Fama-French data
df_combret1 = pd.merge(df_stockret, df_indexret, left_on='MthEnd', right_on='MthEnd')
df_combret2 = pd.merge(df_combret1, df_ff, left_on='MthEnd', right_on='MthEnd')
df_combret2 = df_combret2.dropna()

# Compute excess returns
df_combret2['StockretRf'] = df_combret2['RET'] - df_combret2['Rf']
df_combret2['IndexretRf'] = df_combret2['MthRet'] - df_combret2['Rf']

# Remove a company with only two observations
df_combret3 = df_combret2.loc[df_combret2['CUSIP'] != '55919410']

print(df_combret3.head())
print(df_combret3.columns)

# Example regression for one company - NVIDIA
# Selecting rows based on condition
TestCUSIPS2 = ['67066G10']
df_combret4 = df_combret3[df_combret3['CUSIP'].isin(TestCUSIPS2)]
print(df_combret4.head())
print(df_combret4.columns)

y = df_combret4['StockretRf']
x = df_combret4[['IndexretRf']]
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
print_model = model.summary()
print(print_model)
print(predictions)

# Regression using polyfit
x1 = df_combret4.IndexretRf
y1 = df_combret4.StockretRf
m, b = np.polyfit(x1, y1, 1)
print("Slope: " + str(round(m, 3)))
print("Coefficient: " + str(round(b, 6)))
plt.plot(x1, y1, 'o')
plt.plot(x, m * x + b, 'red', label='NVIDIA excess returns = {:.4f} + {:.2f} x S&P 400 MidCap excess returns'
         .format(b, m))
plt.legend(loc='upper left', fontsize=6)
plt.xlabel('S&P MidCap 400 returns')
plt.ylabel('NVIDIA excess returns')
plt.xlim([-0.25, 0.25])
plt.ylim([-0.40, 0.90])
plt.savefig(path + 'Chart02 NVIDIA beta.jpg')
plt.show()

# Example regression for 3 companies - NVIDIA, Qualcomm and AMD
TestCUSIPS3 = ['67066G10', '74752510', '00790310']
df_combret5 = df_combret3[df_combret3['CUSIP'].isin(TestCUSIPS3)]


def regress(df_combret5, yvar, xvars):
    y3 = df_combret5['StockretRf']
    x3 = df_combret5['IndexretRf']
    x3 = sm.add_constant(x3)
    model3 = sm.OLS(y3, x3).fit()
    predictions3 = model3.predict(x3)
    r2 = r2_score(y3, predictions3)
    return model3.params, r2


output3 = df_combret5.groupby('CUSIP').apply(regress, yvar='y3', xvars=['x3'])
print(output3)


# Running the regression on all companies
def regressall(df_combret3, yvar, xvars):
    yall = df_combret3['StockretRf']
    xall = df_combret3['IndexretRf']
    xall = sm.add_constant(xall)
    modelall = sm.OLS(yall, xall).fit()
    predictionsall = modelall.predict(xall)
    r2all = r2_score(yall, predictionsall)
    return modelall.params


output = df_combret3.groupby('CUSIP').apply(regressall, yvar='yall', xvars=['xall'])
print(output)

# Regression to show the inverse relationship between estimates of alpha and index exposure
x4 = output.IndexretRf
y4 = output.const
m, b = np.polyfit(x4, y4, 1)
print("Slope: " + str(round(m, 3)))
print("Coefficient: " + str(round(b, 6)))
plt.plot(x4, y4, 'o')
plt.plot(x4, m * x4 + b, 'red', label='Alpha = {:.4f} + {:.4f} x slope on midcap excess returns'
         .format(b, m))
plt.legend(loc='upper left', fontsize=10)
plt.xlabel('Slope on midcap excess returns')
plt.ylabel('Alpha')
plt.xlim([-8.5, 8.5])
plt.ylim([-0.35, 0.35])
plt.savefig(path + 'Chart02 Alpha and beta.jpg')
plt.show()

# Export results to Excel

with pd.ExcelWriter(path + 'Excel02 Beta estimates 20230114.xlsx') as writer:
    output3.to_excel(writer, sheet_name='Coefficients 3')
    output.to_excel(writer, sheet_name='Coefficients All')

