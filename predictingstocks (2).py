#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import yfinance as yf
import os


# In[51]:


msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max")
DATA_PATH = "msft_data.json"

if os.path.exists(DATA_PATH):
    with open(DATA_PATH) as f:
        msft_hist = pd.read_json(DATA_PATH)
else:
    msft = yf.Ticker("MSFT")
    msft_hist = msft.history(period="max")
    
    msft_hist.to_json(DATA_PATH)
    
tsla = yf.Ticker("TSLA")
tsla_hist = tsla.history(period="max")
DATA_PATH = "tsla_data.json"

if os.path.exists(DATA_PATH):
    with open(DATA_PATH) as f:
        tsla_hist = pd.read_json(DATA_PATH)
else:
    tsla = yf.Ticker("TSLA")
    tsla_hist = tsla.history(period="max")
   
    tsla_hist.to_json(DATA_PATH)


# In[52]:


tsla_hist.head(5)


# In[53]:


msft_hist.head(5)


# In[54]:


#sequencial time series data above


# In[55]:


msft_hist.plot.line(y="Close", use_index=True)


# In[7]:


data = msft_hist[["Close"]]
data = data.rename(columns = {"Close": "Actual_Close"})
data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
#rolling is a pandas feature that lets you go across the data frame in this case every 2 rows i.e. the (2)
#lambda takes any number of arguments but only has one expression

data.head()

#NaN for the first row because there was no leakage data that pandas couldve used to predict it
#rn we are using leakage because we want the algorithm to get "used" to this idea of predicting
#a target of 1 means the algorithm is showing the closing price was higher than the day before
#therefore a target of 0 means its show a closing price is lower than the day before
#the entire purpose of target to do just that for example im thinking that when i buy microsoft on monday at $260
#...lets say when I buy at $260 the next day It closes at $255 I'll get a 0 and can manipulate the data as I please 
#...like lets say I want to train it that whenever we have a target of 0 buy 5% of a share the following day at open
#likewise if it goes to $265 I'll get a target of 1 and train my algorithm to sell 5% of a share when you get a 1
#idk if this is profitable, doesnt sound like it due to the frequency of buying and selling but lets see


# In[8]:


msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)
#by shifting our data forward in this case by 1 I think it means our algorithm can predict the next day rather than
#reporting a previous day 
msft_prev.head(5)
#for example the actual close of 1986 3/13 was 0.060980
#the close on 1986 3/14 is now shifted to be 0.060980


# In[ ]:





# msft_prev.head()

# In[9]:


predictors = ["Close", "Volume", "Open", "High", "Low"]
#our variables we'll use to predict
data = data.join(msft_prev[predictors]).iloc[1:]
#our target is in data and we'll be taking our predictor variables and take after past the first row i.e. the one after
#NaN
data.head()
#all the data we now see for each row is from the day before
#like i said before 3-14 data is actually 3-13 data
#using the 3-13 data to predict 3-14 data thats what this sets up and what we want it to do overall!!


# In[10]:


from sklearn.ensemble import RandomForestClassifier #SciKitLearn has RandomForestClassifier function can identify nonlinear relationships in the data
import numpy as np

model = RandomForestClassifier(n_estimators=500, min_samples_split=200, random_state=1)
#initialzing the randomforestclassifer model 
#we set parameters like estimators--individual decision trees we wanna train and avg their results for a final result
#min samples split allows us to not overfit the data and prevents the tree from "tightening" to the data 
#essentially the less tight our tree is to the better i guess--overfitting is bad bc it does well with model data...
#but not test (real world) data
#random state not needed but will guarentee refreshed data


# In[11]:


train = data.iloc[:-100]
test = data.iloc[-100:]
#here we are taking the whole dataset except the last 100 rows
#then we take the last 100 rows
#to evaluate error we will use cross validation
model.fit(train[predictors], train["Target"])
#now we will fit the model to be trained on the predictor variables we set
#accustomed to target


# In[12]:


from sklearn.metrics import precision_score
#function in scikit called precision score that can accurately rate our model
#identifying how many true positives our model predicted so does when the price actually went up and the algorithm
#...said it would go up
#divided by all the times it said it would go up

#we wanna minimize our false positives
preds = model.predict(test[predictors])
#by default the above is a numpy array
preds = pd.Series(preds, index=test.index)
#so we turn it into a pandas series
precision_score(test["Target"], preds)
#real values vs predicted values


# In[13]:


combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)
#combine our datasets
#adding our target with real target to one plot
combined.plot()


# In[14]:


i = 1000
step = 1

train = data.iloc[0:i].copy()
test = data.iloc[i:(i+step)].copy()
model.fit(train[predictors], train["Target"])
preds = model.predict_proba(test[predictors])


# In[15]:


preds = model.predict_proba(test[predictors])[:,1]
preds = pd.Series(preds, index=test.index)
preds[preds > 0.6] = 1
preds[preds <= 0.6] = 0

preds.head()


# In[16]:


#backtesting
predictions = []

#for loop over datset in increments

for i in range(1000, data.shape[0], step):
    #split into train and test sets
    train = data.iloc[0:i].copy()
    test = data.iloc[i:(i+step)].copy()
    
    #fit the random forest model
    model.fit(train[predictors], train["Target"])
    
    #make predictions
    preds = model.predict_proba(test[predictors])[:,1]
    preds = pd.Series(preds, index=test.index)
    preds[preds > 0.6] = 1
    preds[preds <= 0.6] = 0
    #if the prediction is right by .6 we give it a 1 
    #if the prediction is wrong by 0.6 or equal it goes a 0
    
    #combine predictions and test values
    combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
    
    predictions.append(combined)
    
    predictions[0].head()


# In[17]:


def backtest(data, mdoel, predictors, start= 1000, step=1):
    predictions = []
    #loop oer the datasetin increments
    for i in range(start, data.shape[0], step):
        #split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        #fit the random forest model
        model.fit(train[predictors], train["Target"])
        
        #Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > 0.6] = 1
        preds[preds <= 0.6] = 0
        
        #combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
        
        predictions.append(combined)
        
    return pd.concat(predictions)


# In[18]:


predictions = backtest(data, model, predictors)


# In[19]:


predictions["Predictions"].value_counts()


# In[20]:


predictions["Target"].value_counts()


# In[21]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[22]:


weekly_mean = data.rolling(7).mean()["Close"]
quarterly_mean = data.rolling(90).mean()["Close"]
annual_mean = data.rolling(365).mean()["Close"]


# In[23]:


weekly_trend = data.shift(1).rolling(7).sum()["Target"]


# In[24]:


data["weekly_mean"] = weekly_mean / data["Close"]
data["quarterly_mean"] = quarterly_mean / data["Close"]
data["annual_mean"] = annual_mean / data["Close"]


# In[25]:


data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]


# In[26]:


data["weekly_trend"] = weekly_trend


# In[27]:


data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]


# In[28]:


full_predictors = predictors + ["weekly_mean", 'quarterly_mean', "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio"]


# In[29]:


predictions = backtest(data.iloc[365:], model, full_predictors)


# In[30]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[31]:


#show how many day trades we would make
predictions["Predictions"].value_counts()


# In[32]:


predictions.iloc[-100:].plot()


# In[ ]:


#more predictors to add like activity monitor at times of day overall activity
#economic indicators
#company milestones
#dividends

