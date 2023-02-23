#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import scipy as sp
from scipy.linalg import solve
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as Poly
from scipy.interpolate import interpolate


# In[57]:


import yfinance as yf
import os
import datetime
import pandas_datareader as data
from pandas import Series, DataFrame


# In[48]:


apple = yf.Ticker("AAPL")
apple_hist = apple.history(period="max")
data_path = "aapl_data.json"

if os.path.exists(data_path):
    with open(data_path) as f:
        apple_hist = pd.read_json(data_path)
else:
    apple = yf.Ticker("AAPL")
    apple_hist = apple.history(period="max")
    apple_hist.to_json(data_path)


# In[49]:


apple_hist


# In[50]:


apple_hist.tail(5)


# In[51]:


apple_hist.tail(30).plot.line(y="Close", use_index=True)


# In[52]:


start_date = '2022-01-01'
end_date = '2023-01-01'

symbol = 'AAPL'
apple_data = yf.download(symbol, start_date, end_date)


# In[53]:


apple_data


# In[87]:


apple_data.iloc[:1,4]


# In[120]:


apple_data.loc['2022-01-03']


# In[145]:


apple_data.iloc[:1,5]


# In[121]:


apple_data.columns = apple_data.columns.to_flat_index()
apple_data.columns = [x[1] for x in apple_data.columns]
apple_data.reset_index(inplace=True)
print(apple_data.head())


# In[130]:


apple_data.tail(10)


# In[214]:


dates = pd.Series(["2022-12-16","2022-12-19","2022-12-20","2022-12-21","2022-12-22","2022-12-23","2022-12-27","2022-12-28","2022-12-29","2022-12-30"])


# In[299]:


dates_a = pd.Series([1,2,3,4,5,6,7,8,9,10])


# In[215]:


close_prices = pd.Series(apple_data.iloc[241:,5])


# In[283]:


closing_prices = apple_data.iloc[241,5], apple_data.iloc[242,5], apple_data.iloc[243,5], apple_data.iloc[244,5], apple_data.iloc[245,5], apple_data.iloc[246,5], apple_data.iloc[247,5], apple_data.iloc[248,5], apple_data.iloc[249,5], apple_data.iloc[250,5]


# In[284]:


print(closing_prices)


# In[285]:


apple_data_frame = pd.DataFrame({'dates': dates, 'close_price':closing_prices})

apple_data_frame


# In[286]:


plt.figure(figsize=(12,10))
plt.scatter(apple_data_frame.dates, apple_data_frame.close_price)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[301]:


apple_dataF = pd.DataFrame({"dates": dates_a, 'close_price':closing_prices})

apple_dataF


# In[305]:


plt.figure(figsize=(12,10))
plt.scatter(apple_dataF.dates, apple_dataF.close_price)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[306]:


vandermonde_matrix = np.vander(dates_a)


# In[307]:


print(vandermonde_matrix)


# In[309]:


coefficients = sp.linalg.solve(vandermonde_matrix, closing_prices)
print(coefficients, "---coefficients we get from solving the vandermonde of dates with values of close price")


# In[311]:


polynomial_interpolation = Poly(coefficients[::-1])
print(polynomial_interpolation, "---our official polynomial post interpolation")
#we get a warning our problem is ill conditioned and accuracy is not storng


# In[321]:


plt.title("interpolation of Apple History Data using vandermonde Matrix")
plt.scatter(apple_dataF.dates, apple_dataF.close_price)
plt.xlabel("History (Days)")
plt.ylabel("Closing Price($)")
x = dates_a
y_vandermonde = polynomial_interpolation(dates_a)
plt.plot(x,y_vandermonde, "r")
plt.show()


# In[ ]:


#Next would be adding more data points and history data and seeing what it does to the curve & points


# In[ ]:


#Evaluating using Newton polynomial interpolation 


# In[ ]:


#Evaluating using Legrange polynomial interpolation


# In[267]:


apple_data.iloc[:2,5]


# In[268]:


apple_data.iloc[0,:]


# In[269]:


apple_data.iloc[241:,5]


# In[270]:


print(dates)
print(close_prices)


# In[271]:


print(apple_data_frame)


# In[252]:


apple_data.interpolate


# In[256]:





# In[ ]:




