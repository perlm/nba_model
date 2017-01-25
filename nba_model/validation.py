import pandas as pd
from sklearn import metrics
from ggplot import *

#################
# validate! - go through past predictions and see how they're performing!
#################

def validate():
	filename = "{}/nba_model/data/games_2017.csv".format(os.path.expanduser("~"))
	actuals = pd.read_csv(filename,delimiter=',',header=0)
	filename = '{}/nba_model/data/predictions.csv'.format(os.path.expanduser("~"))
	predictions = pd.read_csv(filename,delimiter=',',header=0)
	
	allData = pd.merge(left=predictions,right=actuals,how='left',on=['home_team','away_team','date'])

	df = allData.loc[(allData['winner_y'].isin([0,1]))]	
	df['prediction_class'] = 0
	df.loc[df['prediction'] >= 0.5, 'prediction_class'] = 1
	
	#print(metrics.classification_report(df['winner_y'], df['prediction_class']))
	accuracy = metrics.accuracy_score(df['winner_y'], df['prediction_class'])
	auc = metrics.roc_auc_score(df['winner_y'], df['prediction_class'])
	preds = len(df)
	maxDate = max(allData['date'].loc[(allData['winner_y'].isin([0,1]))])

	df = df.round({'prediction': 1})
	forcalibration = df['winner_y'].groupby(df['prediction']).mean().reset_index()
	
	p = ggplot(forcalibration,aes(x="prediction",y="winner_y")) + \
	geom_point(size=100) + \
	xlab('Predicted Win Probability') +	ylab('Actual Win Frequency') + \
	geom_abline(intercept=0,slope=1) + \
	scale_y_continuous(limits=[0,1])+scale_x_continuous(limits=[0,1]) +\
	ggtitle('NBA Prediction Validation.\nDate:{0} Samples:{1}\nAUC:{2} ACC:{3}'.format(maxDate,preds,round(auc,2),round(accuracy,2)))
	
	#ggsave(p, "{}/nba_model/docs/validation_plot.png".format(os.path.expanduser("~")))
	p.save("{}/nba_model/docs/validation_plot.png".format(os.path.expanduser("~")))
	
if __name__ == '__main__':
    validate()


