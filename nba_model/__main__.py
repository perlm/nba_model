from .getData import *
from .buildModel import *
from .tweetIt import *
from .validation import *
import datetime, os


#######################
# This is the master script which will call functions from the other scripts.
# Currently setup to get new data, scrape, model, and publish.
# set to run on cron.
########################


def main():
	if not os.path.isdir('{}/nba_model/data/'.format(os.path.expanduser("~"))):os.makedirs('{}/nba_model/data'.format(os.path.expanduser("~")))

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

	####################
	# pull in data for predictions.
	df_predict = readRawPredictionFile()
	df = addrows(df,df_predict)
	X, X_scaled, Y, scaler,X_fix = processData(df,scaler)
	y_probs_lr = predict(X_scaled,model_lr)

	##########
	# tweet it!
	if len(df.loc[(df['date']==datetime.date.today())])>0:
		tweetProb(extract_new_predictions(df,y_probs_lr))
		store_predictions(df,y_probs_lr)

	##########
	# validate previous predictions!
	validate()


if __name__ == "__main__":
	main()
