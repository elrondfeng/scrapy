# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tweepy
import time
import seaborn as sns

# Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "4M7STdbzUHdB7VdiRcG8Wxt7y"
consumer_secret = "oD8wPy5e1fj5Wmz7DmQRpynNARjt8EKjhRMZlTfwwtpCe0lvp5"
access_token = "525473648-jSba2sYnEIXAKmBrQ9yJxacxWjEuROmHGRBCb061"
access_token_secret = "hhn0Wnu40oEh6XFcIW4cwYmJI3HvXa7pDUvfgQ2PiA5Pn"

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target Account
target_user = "@DukeEnergy"

# Counter
counter = 1

# Variables for holding sentiments
sentiments = []

# Loop through 50 pages of tweets (total 1000 tweets)
for x in range(50):

    # Get all tweets from home feed
    public_tweets = api.user_timeline(target_user)

    # Loop through all tweets
    for tweet in public_tweets:
        # Print Tweets
        # print("Tweet %s: %s" % (counter, tweet["text"]))

        # Run Vader Analysis on each tweet
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = counter

        # Add sentiments for each tweet into an array
        sentiments.append({"Date": tweet["created_at"],
                           "Compound": compound,
                           "Positive": pos,
                           "Negative": neu,
                           "Neutral": neg,
                           "Tweets Ago": counter})

        # Add to counter
        counter = counter + 1

# Convert sentiments to DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd.head()


# Create plot
plt.plot(np.arange(len(sentiments_pd["Compound"])),
         sentiments_pd["Compound"], marker="o", linewidth=0.5,
         alpha=0.8)

# # Incorporate the other graph properties
plt.title("Sentiment Analysis of Tweets (%s) for %s" % (time.strftime("%x"), target_user))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.show()