#!/usr/bin/python

##
# This is the master script which will call functions from the other scripts.
# Currently setup to get new data, scrape, model, and publish.
# will set to run on cron.
## 

from getData import *
from buildModel import *
from tweetIt import *
import datetime

def main():

	###############
	# scrape data!
	#get_all_data_for_modeling()	# if starting from scratch
	get_new_data_for_modeling()	# if want all available training data
	get_data_for_predicting()	# necessary

	###############
	# make model!
	df = readRawFiles()
	X, X_scaled, Y, scaler,X_fix = processData(df)
	model_lr = buildLogisticModel(X_scaled,Y,X_fix,optimize=False)
	#y_probs_lr = predict(X_scaled,model_lr)

	##########
	# pull in data for predictions
	df_predict = readRawPredictionFile()
	df = addrows(df,df_predict)
	X, X_scaled, Y, scaler,X_fix = processData(df,scaler)
	y_probs_lr = predict(X_scaled,model_lr)
	new_df = extract_new_predictions(df,y_probs_lr)


	# tweet it!
	# check length?
	if len(new_df)>0:
		tweetProb(new_df)
		store_predictions(df,y_probs_lr)



if __name__ == "__main__":
	main()
