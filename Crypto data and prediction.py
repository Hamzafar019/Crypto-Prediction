#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries and modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
import yfinance as yf
import pandas as pd

# Get historical price data for Bitcoin and clean it up
btc_ticker = yf.Ticker("BTC-USD")
btc = btc_ticker.history(period="max")
btc.index = pd.to_datetime(btc.index)
btc.index = btc.index.tz_localize(None)
del btc["Dividends"]
del btc["Stock Splits"]
btc.columns = [c.lower() for c in btc.columns]

# Plot the closing prices over time
btc.plot.line(y="close", use_index=True)

# Read in the Wikipedia edits data saved locally and merge it with the Bitcoin data
wiki = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
btc = btc.merge(wiki, left_index=True, right_index=True)

# Create a binary target variable based on whether the price goes up or down tomorrow
btc["tomorrow"] = btc["close"].shift(-1)
btc["target"] = (btc["tomorrow"] > btc["close"]).astype(int)
btc["target"].value_counts()


#base model
# Train a Random Forest classifier on the data and evaluate its performance
# model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
# train = btc.iloc[:-200]
# test = btc[-200:]
# predictors = ["close", "volume", "open", "high", "low", "edit_count", "sentiment", "neg_sentiment"]
# model.fit(train[predictors], train["target"])
# preds = model.predict(test[predictors])
# preds = pd.Series(preds, index=test.index)
# precision_score(test["target"], preds)

# Define a function to predict the target variable given a set of predictors and a trained model
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="predictions")
    combined = pd.concat([test["target"], preds], axis=1)
    return combined

# Define a function to perform a backtest on the model
def backtest(data, model, predictors, start=1095, step=150):
    all_predictions = []
    
    # Iterate over time periods of the data and make predictions
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    # Combine all predictions into a single DataFrame and return it
    return pd.concat(all_predictions)

# Train an XGBoost classifier using backtesting and evaluate its performance
model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200)
predictions = backtest(btc, model, predictors)
precision_score(predictions["target"], predictions["predictions"])

# Define a function to compute rolling averages and ratios of various features
def compute_rolling(btc):
    horizons = [2, 7, 60, 365]
    new_predictors = ["close", "sentiment", "neg_sentiment"]
    
    
    # calculate rolling features for each horizon and add them to the dataframe
    for horizon in horizons:
        # calculate the rolling average of "close" over the given horizon
        rolling_averages=btc.rolling(horizon, min_periods=1).mean()

        # create a new column name for the rolling ratio of "close" to its rolling average
        ratio_column=f"close_ratio_{horizon}"

        # calculate the rolling ratio of "close" to its rolling average and add it to the dataframe
        btc[ratio_column]=btc["close"]/rolling_averages["close"]

        # create a new column name for the rolling average of "edit_count"
        edit_column=f"edit_{horizon}"

        # calculate the rolling average of "edit_count" over the given horizon and add it to the dataframe
        btc[edit_column]=rolling_averages["edit_count"]

        # calculate the rolling trend of "target" over the given horizon and add it to the dataframe
        rolling=btc.rolling(horizon, closed="left", min_periods=1).mean()
        trend_column=f"trend_{horizon}"
        btc[trend_column]=rolling["target"]

        # add the newly created columns to the list of new predictors
        new_predictors+=[ratio_column, trend_column, edit_column]

    # return the modified dataframe and the list of new predictors
    return btc, new_predictors
# Call the function compute_rolling to add the rolling features to the btc dataframe and get the new list of predictors.
btc, new_predictors = compute_rolling(btc.copy())
# Use the XGBClassifier model and the backtest function to make predictions using the new list of predictors 
# and calculate the precision score.
precision_score(predictions["target"], predictions["predictions"]) # output the precision score

