from .getData import *
from .buildModel import *
from .tweetIt import *
from .validation import *
import datetime, os, boto3,bz2,pickle


#######################
# This is the master script which will call functions from the other scripts.
# Currently setup to get new data, scrape, model, and publish.
# set to run on cron.

# moving to doing the scraping from lambda, so it's accessible to the heroku site.
# need to improve the cloudiness of the local runs,
# which I still want to do for performance evaluation, tweeting, etc.
########################


def main():
        s3 = boto3.client('s3')


	if not os.path.isdir('{}/nba_model/data/'.format(os.path.expanduser("~"))):os.makedirs('{}/nba_model/data'.format(os.path.expanduser("~")))

	###############
	# scrape data!
	#get_all_data_for_modeling()	# if starting from scratch
	#get_new_data_for_modeling()	# if want all available training data
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
	X, X_scaled, Y, nba_scaler,X_fix = processData(df,scaler)

	# put this scaler object on aws so the heroku page can access
        #BUCKET_NAME = 'jeopardydata' # replace with your bucket name
        #with bz2.BZ2File("{}/nba_model/model_pickles/nba_scaler.pickle".format(os.path.expanduser("~")),"w") as f:
        #        pickle.dump(nba_scaler, f)
        #s3 = boto3.resource('s3')
        #s3.meta.client.upload_file("{}/nba_model/model_pickles/nba_scaler.pickle".format(os.path.expanduser("~")), BUCKET_NAME, 'nba_scaler.pickle')


	y_probs_lr = predict(X_scaled,model_lr)

	##########
	# tweet it!
	if len(df.loc[(df['date']==datetime.date.today())])>0:
		tweetProb(extract_new_predictions(df,y_probs_lr))
		store_predictions(df,y_probs_lr)

	##########
	# validate previous predictions - need to change to all predictions
	validate()


if __name__ == "__main__":
	main()
