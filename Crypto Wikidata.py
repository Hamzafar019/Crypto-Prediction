#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary modules
import pandas as pd                # For data analysis and manipulation
import mwclient                    # For interacting with MediaWiki API
import time                        # For time-related operations
from statistics import mean        # For calculating mean of a list
from transformers import pipeline  # For natural language processing
from datetime import datetime      # For working with timestamps

# Connect to the Wikipedia site
site = mwclient.Site("en.wikipedia.org")

# Retrieve the page for "Bitcoin"
page = site.pages["Bitcoin"]

# Get all the revisions of the page
revs = list(page.revisions())

# Sort the revisions by timestamp
revs = sorted(revs, key=lambda rev: rev["timestamp"])

# Initialize a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Define a function to find the sentiment of a given text
def find_sentiment(text):
    # Only analyze the first 250 characters of the text
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "NEGATIVE":
        score *= -1   # Negate the score if the sentiment is negative
    return score

# Initialize a dictionary to store the edits by date
edits = {}

# Loop through each revision and calculate its sentiment
for rev in revs:
    # Get the date of the revision
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    
    # If the date is not in the edits dictionary, add it
    if date not in edits:
        edits[date] = dict(sentiments=list(), edit_count=0)
    
    # Add the sentiment score to the corresponding date's entry in the dictionary
    comment = rev.get("comment", "")
    edits[date]["sentiments"].append(find_sentiment(comment))
    
    # Increment the edit count for the corresponding date
    edits[date]["edit_count"] += 1

# Convert the dictionary to a DataFrame and set the index to dates
edits_df = pd.DataFrame.from_dict(edits, orient="index")

# Convert the index to a datetime format
edits_df.index = pd.to_datetime(edits_df.index)

# Create a date range from the start of Bitcoin's page creation to today
dates = pd.date_range(start="2009-03-08", end=datetime.today())

# Reindex the DataFrame to fill any missing dates with zeros
edits_df = edits_df.reindex(dates, fill_value=0)

# Calculate the rolling 30-day average of the sentiment scores and drop any rows with missing values
rolling_edits = edits_df.rolling(30).mean()
rolling_edits = rolling_edits.dropna()

# Save the rolling average to a CSV file
rolling_edits.to_csv("wikipedia_edits.csv")

