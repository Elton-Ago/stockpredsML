#!/usr/bin/env python
# coding: utf-8

# In[444]:


import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import config as cf
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream
import numpy as np
import pandas as pd


# In[445]:


base_urL= "https://paper-api.alpaca.markets"
base_url1 = "https://data.alpaca.markets"
alpaca_api_key = "PK2VVCQIOCPEDH6OSTTS"
alpaca_secret_key = "depW27g3IjSwAJEmlmZbH5Y1Vk4JY7UZCFingw54"


# In[446]:


api = tradeapi.REST(key_id = alpaca_api_key, secret_key = alpaca_secret_key, base_url = base_urL, api_version = 'v2')


# In[447]:


account = api.get_account()


# In[448]:


print(account.id, account.equity, account.status)


# In[449]:


api.get_bars("AAPL", TimeFrame.Hour, "2020-01-01","2022-10-21",adjustment='raw').df


# In[450]:


def process_bar(bar):
    print(bar)#process bar
    
bar_iter=api.get_bars_iter("AAPL",TimeFrame.Hour,"2020-01-01","2022-10-21",adjustment='raw')
for bar in bar_iter:
    process_bar(bar)


# In[451]:


apple_data = api.get_bars("AAPL", TimeFrame.Day,"2020-01-01","2022-10-21",adjustment='raw').df
print(apple_data.head())


# In[452]:


apple_data.columns = apple_data.columns.to_flat_index()
apple_data.columns = [x[1] for x in apple_data.columns]
apple_data.reset_index(inplace=True)
print(apple_data.head())


# In[453]:


plot = apple_data.plot(x="timestamp",y="l",legend=False)
plot.set_xlabel("Date")
plot.set_ylabel("Apple Close Price($): ")
plt.show()


# In[454]:


#calculating moving averages

apple_data['20_SMA'] = apple_data['l'].rolling(window=20, min_periods=1).mean()
apple_data['10_SMA'] = apple_data['l'].rolling(window=10, min_periods=1).mean()


# In[455]:


#finding crossover points
apple_data["Cross"] = 0.0
apple_data["Cross"] = np.where(apple_data['10_SMA'] > apple_data['20_SMA'], 1.0,0.0)
apple_data['Signal'] = apple_data['Cross'].diff()


# In[456]:


#map numbers to words
map_dict = {-1.0: "sell", 1.0: 'buy', 0.0: 'none'}
apple_data['Signal'] = apple_data['Signal'].map(map_dict)


# In[457]:


#print(apple_data[apple_data['Signal'] != "none"].dropna())
apple_data.plot(x='timestamp',y=["l","20_SMA","10_SMA"],color=["k",'b','m'])
plt.plot(apple_data[apple_data['Signal']=='buy']['timestamp'],apple_data['20_SMA'][apple_data['Signal']=='buy'],"^",markersize=8,color='g',label='buy')

plt.plot(apple_data[apple_data['Signal']=='sell']['timestamp'],apple_data['20_SMA'][apple_data['Signal']=='sell'],'v',markersize=8,color='r',label='sell')

plt.xlabel("Date")
plt.ylabel("Apple Close Price ($)")
plt.legend()
plt.show()


# In[482]:


api.get_bars("TSLA", TimeFrame.Hour, "2021-01-01","2022-10-24",adjustment='raw').df


# In[483]:


tesla_data = api.get_bars("TSLA", TimeFrame.Day,"2021-01-01","2022-10-24",adjustment='raw').df
print(tesla_data.head())


# In[484]:


tesla_data.columns = tesla_data.columns.to_flat_index()
tesla_data.columns = [x[1] for x in tesla_data.columns]
tesla_data.reset_index(inplace=True)
print(tesla_data.head())


# In[485]:


plot = tesla_data.plot(x="timestamp",y="l",legend=False)
plot.set_xlabel("Date")
plot.set_ylabel("Tesla Close Price($): ")
plt.show()


# In[486]:


#calculating moving averages

tesla_data['20_SMA'] = tesla_data['l'].rolling(window=20, min_periods=1).mean()
tesla_data['10_SMA'] = tesla_data['l'].rolling(window=10, min_periods=1).mean()


# In[487]:


#finding crossover points
tesla_data["Cross"] = 0.0
tesla_data["Cross"] = np.where(tesla_data['10_SMA'] > tesla_data['20_SMA'], 1.0,0.0)
tesla_data['Signal'] = tesla_data['Cross'].diff()


# In[488]:


#map numbers to words
map_dict = {-1.0: "sell", 1.0: 'buy', 0.0: 'none'}
tesla_data['Signal'] = tesla_data['Signal'].map(map_dict)


# In[489]:


#print(apple_data[apple_data['Signal'] != "none"].dropna())
tesla_data.plot(x='timestamp',y=["l","20_SMA","10_SMA"],color=["k",'b','m'])
plt.plot(tesla_data[tesla_data['Signal']=='buy']['timestamp'],tesla_data['20_SMA'][tesla_data['Signal']=='buy'],"^",markersize=8,color='g',label='buy')

plt.plot(tesla_data[tesla_data['Signal']=='sell']['timestamp'],tesla_data['20_SMA'][tesla_data['Signal']=='sell'],'v',markersize=8,color='r',label='sell')

plt.xlabel("Date")
plt.ylabel("Tesla Close Price ($)")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[458]:


api.submit_order("AAPL",1,"buy","market",'day')


# In[459]:


api.submit_order("TSLA",1,"sell","market","day")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[462]:


portfolio = api.list_positions()

for position in portfolio:
    print("{} shares of {}".format(position.qty,position.symbol))


# In[463]:


api.get_quotes("APPL",TimeFrame.Day,"2020-06-08","2021-06-08",limit=10).df


# In[464]:


def process_quote(quote):
    print(quote)


# In[465]:


quote_iter=api.get_quotes_iter("APPL","2021-06-08","2021-06-08",limit=10)
for quote in quote_iter:
    process_quote(quote)


# In[466]:


api.get_trades("APPL","2021-06-08","2021-06-08",limit=10).df


# In[467]:


def process_trade(trade):
    print(trade)


# In[468]:


trades_iter = api.get_trades_iter("APPL","2021-06-08","2021-06-08",limit=10)
for trade in trades_iter:
    process_trade(trade)


# In[469]:


async def trade_callback(t):
    print('trade',t)
    
async def quote_callback(q):
    print("quote",q)
    
#class instance
stream = Stream(alpaca_api_key,alpaca_secret_key,base_url=URL("https://paper-api.alpaca.markets"), data_feed='iex')

stream.subscribe_trades(trade_callback,"APPL")
stream.subscribe_quotes(quote_callback,"IBM")

stream.run()


# In[ ]:




