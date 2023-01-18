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
path = 'F:\\UM\Courses\\FIN427\\FIN427 Winter 2023\\'

# Import monthly returns data
df_returns = pd.read_excel(path + 'Excel02 Invesco MidCap Quality 20230104.xlsx', sheet_name='MthData')
print(df_returns.head())
df_returns = df_returns.dropna()
print(df_returns.head())

# Regression of Invesco excess returns on S&P 400 excess returns

y = df_returns['InvescoMidQualExRet']
x = df_returns[['SP400ExRet']]
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
print_model = model.summary()
print(print_model)
print(predictions)

# Regression using polyfit
x1 = df_returns.SP400ExRet
y1 = df_returns.InvescoMidQualExRet
m, b = np.polyfit(x1, y1, 1)
print("Slope: " + str(round(m, 3)))
print("Coefficient: " + str(round(b, 6)))
plt.plot(x1, y1, 'o')
plt.plot(x, m * x + b, 'red', label='Invesco excess returns = {:.4f} + {:.2f} x S&P 400 MidCap Excess returns'
         .format(b, m))
plt.legend(loc='lower right')
plt.xlabel('S&P MidCap 400 excess returns')
plt.ylabel('Invesco excess returns')
plt.xlim([-0.25, 0.25])
plt.ylim([-0.25, 0.25])
plt.savefig(path + 'Invesco scatter vs S&P 400.jpg')
plt.show()
