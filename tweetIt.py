#!/usr/bin/python

import ConfigParser, datetime, twitter, sys, os
import pandas as pd


def tweetProb(df):
    #####################################
    # need to split this into multiple tweets
    
    config = ConfigParser.ConfigParser()
    config.read('/home/jason/.python_keys.conf')
    t = twitter.Twitter(auth=twitter.OAuth(token=config.get('twitter','token'), token_secret=config.get('twitter','token_secret'), consumer_key=config.get('twitter','consumer_key'), consumer_secret=config.get('twitter','consumer_secret')))
    
    tweetstart = '#NBA home team win probs:\n'
    tweetend = 'https://hastydata.wordpress.com/2017/01/21/nba-win-model/'
    
    tweet = tweetstart
    for index, row in df.iterrows():
        tweet += row['away'] + ' at ' + row['home'] + ': ' + row['prediction'] +"\n"

        if len(tweet)>80:
		tweet += tweetend
	        print "Tweet Sent:\n%s" % (str(tweet))
	        t.statuses.update(status=tweet)
		tweet = tweetstart

    if len(tweet)>len(tweetstart):
	tweet += tweetend
	print "Tweet Sent:\n%s" % (str(tweet))
	t.statuses.update(status=tweet)


    

def store_predictions(df,pred):
    ####
    # store predictions for future evaluation!
    ######
    fil = 'data/predictions.csv'
    
    df['date'] = pd.to_datetime(df['date'])
    df['prediction'] = pred
    df = df.loc[(df['date']==datetime.date.today())]

    if (os.path.exists(fil)):
        prev = pd.read_csv(fil,delimiter=',',header=0)
        prev['date'] = pd.to_datetime(prev['date'])
        
        # check if this date is already in there    
        # the max of a datetime column is a timestamp object.
        if datetime.datetime.strptime(str(prev['date'].max())[:10],"%Y-%m-%d").date()<datetime.datetime.today().date():
            df.to_csv(fil, header=False, index=False,mode='a')
    else:        
        df.to_csv(fil, header=True, index=False)



if __name__ == '__main__':
	tweetProb(new_df)
	store_predictions(df,y_probs_lr)


