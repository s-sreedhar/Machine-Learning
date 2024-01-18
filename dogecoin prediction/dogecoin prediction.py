import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor

#Loading the dataset
data = pd.read_csv("doge-usd.csv") 
data.head() 

#checking correlation
data.corr()

data['Date'] = pd.to_datetime(data['Date'], 
                              infer_datetime_format=True) 
data.set_index('Date', inplace=True) 
  
data.isnull().any() 

#converting the string data and time in proper date & time format
data['Date'] = pd.to_datetime(data['Date'], 
                              infer_datetime_format=True) 
data.set_index('Date', inplace=True) 
  
data.isnull().any() 

data.isnull().sum() 

#drop null values
data = data.dropna()

data.describe() 

#analyse the closing price
plt.figure(figsize=(20, 7)) 
x = data.groupby('Date')['Close'].mean() 
x.plot(linewidth=2.5, color='b') 
plt.xlabel('Date') 
plt.ylabel('Volume') 
plt.title("Date vs Close of 2021") 


data["gap"] = (data["High"] - data["Low"]) * data["Volume"] 
data["y"] = data["High"] / data["Volume"] 
data["z"] = data["Low"] / data["Volume"] 
data["a"] = data["High"] / data["Low"] 
data["b"] = (data["High"] / data["Low"]) * data["Volume"] 
abs(data.corr()["Close"].sort_values(ascending=False)) 

data = data[["Close", "Volume", "gap", "a", "b"]] 
data.head() 

#prepring training and testing datasets

df2 = data.tail(30) 
train = df2[:11] 
test = df2[-19:] 
  
print(train.shape, test.shape) 

#model development
from statsmodels.tsa.statespace.sarimax import SARIMAX 
model = SARIMAX(endog=train["Close"], exog=train.drop( 
    "Close", axis=1), order=(2, 1, 1)) 
results = model.fit() 
print(results.summary()) 

#observing the prediction in time series
start = 11
end = 29
predictions = results.predict( 
    start=start, 
    end=end, 
    exog=test.drop("Close", axis=1)) 
predictions 

#plotting
test["Close"].plot(legend=True, figsize=(12, 6)) 
predictions.plot(label='TimeSeries', legend=True) 