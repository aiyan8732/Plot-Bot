

```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Term
request_target_term = "@aiyan8732 Analyze:"
# list to store media_account
media_list = []
```


```python
def PlotBot():
    public_tweets = api.search(request_target_term, count=5, result_type="recent")   
    
    for tweet in public_tweets["statuses"]:
        # locate media account name requested by user
        media_index = tweet["text"].index('@', 1)
        media_account = tweet["text"][media_index:]        
        # request user id
        tweet_id = tweet["id"]
        tweet_author = tweet["user"]["screen_name"]
        # Variables for holding sentiments
        sentiments = []
        # Counter
        counter = 1
        # Variable for max_id
        oldest_tweet = None
        
        # Loop through 25 pages of tweets (total 500 tweets)
        for x in range(25):
            media_tweets = api.user_timeline(media_account, max_id = oldest_tweet)
            for tweet in media_tweets:
                results = analyzer.polarity_scores(tweet["text"])
                compound = results["compound"]
                tweets_ago = counter
                # Get Tweet ID, subtract 1, and assign to oldest_tweet
                oldest_tweet = tweet['id'] - 1
                sentiments.append({"Tweet Account": media_account, 
                                   "Compound": compound,
                                   "Tweets Ago": counter})
                counter += 1
                
        # Convert sentiments to DataFrame
        sentiments_pd = pd.DataFrame.from_dict(sentiments)
        
        # Create plot
        plt.style.use('seaborn-darkgrid')
        x_vals = sentiments_pd["Tweets Ago"]
        y_vals = sentiments_pd["Compound"]
        plt.plot(x_vals,
                 y_vals, marker="o", linewidth=0.5,
                 alpha=0.8, color="blueviolet", label=media_account)
        #  Incorporate the other graph properties
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M")
        plt.title(f"Sentiment Analysis of Tweets ({now}) for {media_account}")
        plt.xlim([x_vals.max()+10,x_vals.min()-10])
        lgd = plt.legend(title="Tweets",loc='lower left', bbox_to_anchor=(1, 0.5))
        plt.ylabel("Tweet Polarity")
        plt.xlabel("Tweets Ago")
        
        # save plot to png
        plt.savefig(f"Scatter_Tweets_Sentiment_{media_account[1:]}.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
        # refresh the graph
        plt.gcf().clear()       

        # only analyzing Twitter accounts that have not previously been stored in media_list
        if media_account not in media_list:
            media_list.append(media_account)
            # Respond to tweet 
            api.update_with_media(f"Scatter_Tweets_Sentiment_{media_account[1:]}.png",
                      f"New Tweet Analysis: {media_account} (Thank you @{tweet_author} !)",in_reply_to_status_id=tweet_id)
            # Print success message
            print("Successful response!")
        else:
            print(f"Please reference previously released sentiment data of {media_account}")
```


```python
while(True):
    PlotBot()
    time.sleep(300)
```

    Successful response!
    Successful response!
    Successful response!
    Please reference previously released sentiment data of @NBA
    Please reference previously released sentiment data of @NFL
    Please reference previously released sentiment data of @MLB



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-4-961dd744573e> in <module>()
          1 while(True):
          2     PlotBot()
    ----> 3     time.sleep(300)
    

    KeyboardInterrupt: 



    <matplotlib.figure.Figure at 0x1101efe80>

