import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_extraction, linear_model, metrics, model_selection, ensemble
import math, glob, datetime, os

####
# This file contains functions for building the classification model.
###

def optimizeLambdaLogistic(X_train, X_test, y_train, y_test,L='l1'):
	# use CV to optimize regularization hyperparameter! (using either L1 or L2) (lambda is inverse C here)

	if L=='l1':
		tuned_parameters = [ {'C':[1e-5,1e-3,5e-3,7.5e-3,1e-2,2.5e-2,5e-2,1e-1,5e-1,1e0,1e8]}]
	else:
		tuned_parameters = [ {'C':[1e-8,1e-6,1e-4,1e-2, 1e0,1e2,1e4,1e6,1e8]}]

	clf = model_selection.GridSearchCV(linear_model.LogisticRegression(penalty=L), tuned_parameters, cv=50,scoring='roc_auc')
	clf.fit(X_train, y_train)

	print "Hyperparameter Optimization, penalty=", L
	print(clf.best_params_)

	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	#for mean, std, params in zip(means, stds, clf.cv_results_['params']):print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	y_prob = clf.predict_proba(X_test)[:,1]
	y_class = clf.predict(X_test)
	#print y_test, y_prob, y_class
	#print "Hyperparameter Optimization"
	#print(metrics.classification_report(y_test, y_class))

	return clf.best_params_


def buildLogisticModel(X_scaled,Y,X_fix,optimize=True):
	# build a model! l1 for lasso, l2 for ridge
	# use CV and holdout.
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, Y, test_size=0.3, random_state=0)

	# need to reshape for some reason...
	Y = Y.as_matrix()
	c, r = Y.shape
	Y = Y.reshape(c,)

	y_train = y_train.as_matrix()
	c, r = y_train.shape
	y_train = y_train.reshape(c,)

	y_test = y_test.as_matrix()
	c, r = y_test.shape
	y_test = y_test.reshape(c,)

	if optimize:
	    # optimize hyperparameter	
	    la = optimizeLambdaLogistic(X_train, X_test, y_train, y_test,'l1')
	    #lb = optimizeLambdaLogistic(X_train, X_test, y_train, y_test,'l2')	# consistently less good.

	    # train model using hyperparameter
	    model = linear_model.LogisticRegression(C=la['C'], penalty='l1')
	else:
	    model = linear_model.LogisticRegression(C=0.05, penalty='l1')
	model.fit(X_train,y_train)

	y_prob = model.predict_proba(X_test)[:,1]
	y_class = model.predict(X_test)
	print "Final Model: Out of Sample Performance"
	print(metrics.classification_report(y_test, y_class))

	print "AUC:", metrics.roc_auc_score(y_test, y_prob)

	# retrain on whole data set.
	if optimize:
		model = linear_model.LogisticRegression(C=la['C'], penalty='l1')
	else:
		model = linear_model.LogisticRegression(C=0.05, penalty='l1')
	model.fit(X_scaled,Y)

	print model.intercept_
	factors = list(X_fix.columns.values)
	coefs 	= list(model.coef_.ravel())
	for i,f in enumerate(factors):
		print f,"\t", coefs[i]

	return model


def buildRandomForest(X_scaled,Y,X_fix):
	########################
	# try a random forest model!
	##########################3
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, Y, test_size=0.3, random_state=0)
    
    # need to reshape for some reason...
    Y = Y.as_matrix()
    c, r = Y.shape
    Y = Y.reshape(c,)
    
    y_train = y_train.as_matrix()
    c, r = y_train.shape
    y_train = y_train.reshape(c,)
    
    y_test = y_test.as_matrix()
    c, r = y_test.shape
    y_test = y_test.reshape(c,)
    
    rf = ensemble.RandomForestClassifier(n_estimators=1000)
    rf.fit(X_train, y_train)
    
    y_prob = rf.predict_proba(X_test)[:,1]
    y_class = rf.predict(X_test)
    print "Final RF Model: Out of Sample Performance"
    print(metrics.classification_report(y_test, y_class))
    print "AUC:", metrics.roc_auc_score(y_test, y_prob)
    
    # retrain on whole data set.
    rf.fit(X_scaled, Y)
    #importances = list(rf.feature_importances_)
    #factors = list(X_fix.columns.values)
    #for i,f in enumerate(factors):
    #    print f,"\t", importances[i]
    return rf




def predict(X_scaled,model):
	# for a given model and independent data, generate predictions
	y_prob = model.predict_proba(X_scaled)[:,1]
	print "Avg of predictions= ", np.mean(y_prob)
	return y_prob


def readRawFiles():
	# read in all csv's and return pandas dataframe
	filenames = glob.glob("{}/nba_model/data/games_*.csv".format(os.path.expanduser("~")))
	df = pd.concat([pd.read_csv(f,delimiter=',',header=0) for f in filenames], ignore_index=True)
	return df
	

def readRawPredictionFile():
	# read in csv's for prediction set and return pandas dataframe
	f = "{}/nba_model/data/predict_games.csv".format(os.path.expanduser("~"))
	df = pd.read_csv(f,delimiter=',',header=0)
	return df

def construct_features(df):
    #################
    # given df with some features, add in some more
    #################
    # optimize feature set
    # auc = 0.724, f1=0.66
    # less features:
    # 724, 0.67. 66 with days off joint
    
    # Add features!
    # try bucketing instead of linear assumption for daysoff
    # df['winner'].groupby(df['daysoff_home']).mean() # 1 day is good. more if inconsequential
    df['daysoff_home_bucket'] = pd.cut(df['daysoff_home'],bins=[-0.5,0.5,1.5,100],labels=['a_0','b_1','c_gt1'])
    df['daysoff_away_bucket'] = pd.cut(df['daysoff_away'],bins=[-0.5,0.5,1.5,100],labels=['a_0','b_1','c_gt1'])
    
    df['daysoff_diff'] = df['daysoff_home'] - df['daysoff_away']
    df['daysoff_diff_bucket'] = pd.cut(df['daysoff_diff'],bins=[-100,-1.5,-0.5,0.5,1.5,100],labels=['a_away_2plus','b_away_1','c_same','d_home_1','e_home_2plus'])
    
    df['cumpointdiff_diff'] = df['cumpointdiff_home'] - df['cumpointdiff_away']
    df['winrate_diff'] = df['winrate_home'] - df['winrate_away']
    df['win10_diff'] = df['win10_home']-df['win10_away']

    df['date'] = pd.to_datetime(df['date'])

    # season = starting year
    df['season'] = 2005
    df.loc[((df['date'] > '2006-08-01') & (df['date'] < '2007-08-01')), 'season'] = 2006
    df.loc[((df['date'] > '2007-08-01') & (df['date'] < '2008-08-01')), 'season'] = 2007
    df.loc[((df['date'] > '2008-08-01') & (df['date'] < '2009-08-01')), 'season'] = 2008
    df.loc[((df['date'] > '2009-08-01') & (df['date'] < '2010-08-01')), 'season'] = 2009
    df.loc[((df['date'] > '2010-08-01') & (df['date'] < '2011-08-01')), 'season'] = 2010
    df.loc[((df['date'] > '2011-08-01') & (df['date'] < '2012-08-01')), 'season'] = 2011
    df.loc[((df['date'] > '2012-08-01') & (df['date'] < '2013-08-01')), 'season'] = 2012
    df.loc[((df['date'] > '2013-08-01') & (df['date'] < '2014-08-01')), 'season'] = 2013
    df.loc[((df['date'] > '2014-08-01') & (df['date'] < '2015-08-01')), 'season'] = 2014
    df.loc[((df['date'] > '2015-08-01') & (df['date'] < '2016-08-01')), 'season'] = 2015
    df.loc[((df['date'] > '2016-08-01') & (df['date'] < '2017-08-01')), 'season'] = 2016

    # I want previous season record.
    # I'm going to cheat a bit and use record as of last home game of season for the home team and away for the away team.
    idx = df.groupby(['season','home_team'], sort=False)['date'].transform(max) == df['date']
    temp = df[idx].copy()
    temp['season'] = temp['season']+1
    temp['last_season_home'] = temp['winrate_home']
    df = df.merge(temp[['home_team','season','last_season_home']], how='left',on=['home_team','season'])
    
    idx = df.groupby(['season','away_team'], sort=False)['date'].transform(max) == df['date']
    temp = df[idx].copy()
    temp['season'] = temp['season']+1
    temp['last_season_away'] = temp['winrate_away']
    df = df.merge(temp[['away_team','season','last_season_away']], how='left',on=['away_team','season'])
    # impute (changing team names and prior to first downloaded season).
    # Could manually set to more accurate values
    df['last_season_home'] = df['last_season_home'].fillna(0.5)
    df['last_season_away'] = df['last_season_away'].fillna(0.5)
    df['last_season_diff'] = df['last_season_home']-df['last_season_away']
    
    # should actually calc strength of sched. Use east/west as quick proxy. same since 2005.
    west = ['Utah Jazz','Denver Nuggets','Memphis Grizzlies','Houston Rockets','Minnesota Timberwolves',
        'Portland Trail Blazers','Dallas Mavericks','Golden State Warriors','San Antonio Spurs',
        'Phoenix Suns','Sacramento Kings','Los Angeles Lakers','Los Angeles Clippers','Oklahoma City Thunder',
        'New Orleans Hornets','New Orleans Pelicans','Seattle SuperSonics','New Orleans/Oklahoma City Hornets']
    
    # if home is west and away is east = 1. if home is east and away is west -1. else 0.
    df['east_west'] = 0 
    df.loc[(df['home_team'].isin(west)) & (~df['away_team'].isin(west)), 'east_west'] = 1
    df.loc[(~df['home_team'].isin(west)) & (df['away_team'].isin(west)), 'east_west'] = -1
    
    # distance travelled!
    # lat longs courtesy of http://online-code-generator.com/us-major-sports-dataset.php
    f = "{}/nba_model/data/lat_longs.csv".format(os.path.expanduser("~"))
    latlongs = pd.read_csv(f,delimiter=',',header=0)
    latlongs['home_team'] = latlongs['team_name']
    latlongs['home_team_lat'] = latlongs['lat']
    latlongs['home_team_lon'] = latlongs['lon']
    df = df.merge(latlongs[['home_team','home_team_lat','home_team_lon']], how='left',on=['home_team'])

    latlongs['away_team'] = latlongs['team_name']
    latlongs['away_team_lat'] = latlongs['lat']
    latlongs['away_team_lon'] = latlongs['lon']
    df = df.merge(latlongs[['away_team','away_team_lat','away_team_lon']], how='left',on=['away_team'])
    df['distance'] = haversine(df['home_team_lon'],df['home_team_lat'],df['away_team_lon'],df['away_team_lat'])

    return df



def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

    

def processData(df,scaler=None):
    #################3
	# take dataframe and reformat for sci-kit learn including normalization
    ################
    
    df = construct_features(df)

    # choose features in model! - try including game count, may be handled by rf
    #X = df[['cumpointdiff_away','cumpointdiff_home','daysoff_away_bucket','daysoff_home_bucket','winrate_away','winrate_home','win10_away','win10_home','game_count_away','game_count_home']]
    #X = df[['cumpointdiff_away','cumpointdiff_home','daysoff_away_bucket','daysoff_home_bucket','winrate_away','winrate_home','win10_away','win10_home']]
    #X = df[['cumpointdiff_diff','winrate_diff','win10_diff','daysoff_away_bucket','daysoff_home_bucket','last_season_diff','east_west']]
    X = df[['cumpointdiff_diff','winrate_diff','win10_diff','daysoff_diff_bucket','last_season_diff','east_west']]
    
    # if using multi-level factors need to do this:
    X_fix = pd.get_dummies(X)
    # X_fix = X
    
    Y = df[['winner']]
    if scaler is None:
        print "Win Rate in data= ", Y['winner'].mean()
    
    # scale features!
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(X_fix)	#this allows me to re-use the scaler.
        X_scaled = scaler.transform(X_fix) 
    else:
	    X_scaled = scaler.transform(X_fix) 
    
    return X,X_scaled, Y, scaler, X_fix

def addrows(df,df_predict):  
    return pd.concat([df,df_predict])

def extract_new_predictions(df,y_probs):
    #####################
    # take combined df and pull out just today's games
    ##################3
    df['prediction'] = y_probs
    df['date'] = pd.to_datetime(df['date'])
    #df = df.loc[(df['date']==datetime.date.today())]
    df['away'] = df.away_team.str.split().str.get(-1)
    df['home'] = df.home_team.str.split().str.get(-1)
    df['prediction'] = pd.Series(["{0:.0f}%".format(prediction * 100) for prediction in df['prediction']], index = df.index)

    return df[['away','home','prediction']].loc[(df['date']==datetime.date.today())]
    
    
if __name__ == '__main__':
    df = readRawFiles()
    X, X_scaled, Y, scaler,X_fix = processData(df)
    model_lr = buildLogisticModel(X_scaled,Y,X_fix,optimize=False)
    y_probs_lr = predict(X_scaled,model_lr)
    #model_rf = buildRandomForest(X_scaled,Y,X_fix)


    ##########
    # pull in data for predictions
    df_predict = readRawPredictionFile()
    df = addrows(df,df_predict)
    X, X_scaled, Y, scaler,X_fix = processData(df,scaler)
    y_probs_lr = predict(X_scaled,model_lr)
    new_df = extract_new_predictions(df,y_probs_lr)

    #model_rf = buildRandomForest(X_scaled,Y,X_fix)
    #y_probs_rf = predict(X_scaled,model_rf)


