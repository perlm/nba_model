import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates


import pandas as pd
import numpy as np
from sklearn import metrics
from ggplot import *
import os

#################
# validate! - go through past predictions and see how they're performing!
#################

def validate():
	# matplotlib has multiplot, ggplot doesn't
	pdf = PdfPages("{}/nba_model/docs/validation_plots.pdf".format(os.path.expanduser("~")))
    
	filename = "{}/nba_model/data/games_2017.csv".format(os.path.expanduser("~"))
	actuals = pd.read_csv(filename,delimiter=',',header=0)
	filename = '{}/nba_model/data/predictions.csv'.format(os.path.expanduser("~"))
	predictions = pd.read_csv(filename,delimiter=',',header=0)
	
	allData = pd.merge(left=predictions,right=actuals,how='left',on=['home_team','away_team','date'])

	df = allData.loc[(allData['winner_y'].isin([0,1]))].copy()
	df['prediction_class'] = 0
	df.loc[df['prediction'] >= 0.5, 'prediction_class'] = 1
	
	#print(metrics.classification_report(df['winner_y'], df['prediction_class']))
	accuracy = metrics.accuracy_score(df['winner_y'], df['prediction_class'])
	auc = metrics.roc_auc_score(df['winner_y'], df['prediction_class'])
	preds = len(df)
	maxDate = max(allData['date'].loc[(allData['winner_y'].isin([0,1]))])

	df = df.round({'prediction': 1})
	forcalibration = df['winner_y'].groupby(df['prediction']).mean().reset_index()
	
	plot1=plt.figure(figsize=(7, 5))
	plt.plot(forcalibration['prediction'],forcalibration['winner_y'])
	plt.ylabel('Actual Win Frequency')
	plt.xlabel('Predicted Win Probability')
	plt.title('NBA Prediction Validation.\nDate:{0} Samples:{1}\nAUC:{2} ACC:{3}'.format(maxDate,preds,round(auc,2),round(accuracy,2)))
	#plt.show()
	pdf.savefig(plot1)
	plt.close()
	
	#plot1=ggplot(forcalibration,aes(x="prediction",y="winner_y")) + \
	#geom_point(size=100) + \
	#xlab('Predicted Win Probability') +	ylab('Actual Win Frequency') + \
	#geom_abline(intercept=0,slope=1) + \
	#scale_y_continuous(limits=[0,1])+scale_x_continuous(limits=[0,1]) +\
	#ggtitle('NBA Prediction Validation.\nDate:{0} Samples:{1}\nAUC:{2} ACC:{3}'.format(maxDate,preds,round(auc,2),round(accuracy,2)))
	
	#ggsave(p, "{}/nba_model/docs/validation_plot.png".format(os.path.expanduser("~")))
	#p.save("{}/nba_model/docs/validation_plot.png".format(os.path.expanduser("~")))
	
	#figure 2
	df['correct'] = 0
	df.loc[((df['prediction_class']==1) & (df['winner_y']==1)), 'correct'] = 1
	df.loc[((df['prediction_class']==0) & (df['winner_y']==0)), 'correct'] = 1

	forplot2 = df.groupby(['date']).aggregate({"correct":np.mean})
	forplot2 = forplot2.reset_index()
	forplot2['date'] = pd.to_datetime(forplot2['date'])
    
	#plot2=ggplot(forplot2,aes(x="date",y="correct")) + \
	#geom_point(size=100) + geom_line(size=3) + \
	#xlab('Date') +	ylab('Model Accuracy') + \
	#scale_y_continuous(limits=[0,1])+\
	#scale_x_date(labels='%m/%d/%Y') +\
	#ggtitle('NBA model validation by day')
    
	plot2, ax =plt.subplots(figsize=(7, 5))
	plt.plot(forplot2['date'],forplot2['correct'])
	
	days = mdates.DayLocator()
	ax.xaxis.set_major_locator(days)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	plot2.autofmt_xdate()

	plt.ylabel('Model Accuracy')
	plt.xlabel('Date')
	plt.title('NBA model validation by day')
	#plt.show()
	pdf.savefig(plot2)
	plt.close()

	pdf.close()
	
if __name__ == '__main__':
    validate()


