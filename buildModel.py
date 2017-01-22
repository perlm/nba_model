import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_extraction, linear_model, metrics, model_selection, ensemble
import math, glob, datetime

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
    
    rf = ensemble.RandomForestClassifier(n_estimators=250)
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
	filenames = glob.glob("data/games_*.csv")
	df = pd.concat([pd.read_csv(f,delimiter=',',header=0) for f in filenames], ignore_index=True)
	return df
	

def readRawPredictionFile():
	# read in csv's for prediction set and return pandas dataframe
	f = "data/predict_games.csv"
	df = pd.read_csv(f,delimiter=',',header=0)
	return df

def processData(df,scaler=None):
    #################3
	# take dataframe and reformat for sci-kit learn including normalization
    ################
    
    # optimize feature set
    
    # Add features!
    # try bucketing instead of linear assumption for daysoff
    # df['winner'].groupby(df['daysoff_home']).mean() # 1 day is good. more if inconsequential
    df['daysoff_home_bucket'] = pd.cut(df['daysoff_home'],bins=[-0.5,0.5,1.5,100],labels=['a_0','b_1','c_gt1'])
    df['daysoff_away_bucket'] = pd.cut(df['daysoff_away'],bins=[-0.5,0.5,1.5,100],labels=['a_0','b_1','c_gt1'])
    

    # choose features in model! - try including game count, may be handled by rf
    #X = df[['cumpointdiff_away','cumpointdiff_home','daysoff_away_bucket','daysoff_home_bucket','winrate_away','winrate_home','win10_away','win10_home','game_count_away','game_count_home']]
    X = df[['cumpointdiff_away','cumpointdiff_home','daysoff_away_bucket','daysoff_home_bucket','winrate_away','winrate_home','win10_away','win10_home']]

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
    df = df.loc[(df['date']==datetime.date.today())]
    df['away'] = df.away_team.str.split().str.get(-1)
    df['home'] = df.home_team.str.split().str.get(-1)
    df['prediction'] = pd.Series(["{0:.0f}%".format(prediction * 100) for prediction in df['prediction']], index = df.index)

    return df[['away','home','prediction']]
    
    
if __name__ == '__main__':
    df = readRawFiles()
    X, X_scaled, Y, scaler,X_fix = processData(df)
    model_lr = buildLogisticModel(X_scaled,Y,X_fix,optimize=False)
    y_probs_lr = predict(X_scaled,model_lr)
    
    model_rf = buildRandomForest(X_scaled,Y,X_fix)
    y_probs_rf = predict(X_scaled,model_rf)

    # evaluate with plots for bloggering (plots.py)
    #(i had them here in functions, but rodeo doesn't seem to plot them)

    # pull in data for prediction
    df_predict = readRawPredictionFile()
    df = addrows(df,df_predict)
    X, X_scaled, Y, scaler,X_fix = processData(df,scaler)
    y_probs_lr = predict(X_scaled,model_lr)
    new_df = extract_new_predictions(df,y_probs_lr)
    
