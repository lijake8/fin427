# Import and view data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

# set path to current folder
path = '.'

start = datetime.datetime(1995, 5, 4)
end = datetime.datetime(2022, 12, 20)
spy = yf.download('SPY', start=start, end=end)
mdy = yf.download('MDY', start=start, end=end)
print(mdy.columns)
print(mdy.head())
print(mdy['Adj Close']) # use adjusted price to take into account dividends and splits
mdy['cumret']=mdy['Adj Close']/mdy['Adj Close'][0]
spy['cumret']=spy['Adj Close']/spy['Adj Close'][0]
mdy.to_csv(path + 'mdy.csv', index=True)
spy.to_csv(path + 'spy.csv', index=True)
mdy.to_excel(path + 'mdy.xlsx', index=True, sheet_name='mdy')
spy.to_excel(path + 'spy.xlsx', index=True, sheet_name='spy')

mdy['cumret'].plot(label='MDY', color='cyan')
spy['cumret'].plot(label='SPY', color='purple')

plt.title('Cumulative Return to $1')
plt.xlabel('Date')
plt.legend()
plt.savefig(path + 'returns.jpg')
plt.show()

