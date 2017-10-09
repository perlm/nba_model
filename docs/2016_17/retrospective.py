import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
from sklearn import metrics

# build the model
#	df = readRawFiles()
#	X, X_scaled, Y, scaler,X_fix = processData(df)
#	model_lr = buildLogisticModel(X_scaled,Y,X_fix,optimize=False)
# see the model predictions
# y_probs_lr = predict(X_scaled,model_lr)


df['prediction'] = y_probs_lr
df['prediction_class'] = 0
df.loc[df['prediction'] >= 0.5, 'prediction_class'] = 1
	
accuracy = metrics.accuracy_score(df['winner'], df['prediction_class'])
auc = metrics.roc_auc_score(df['winner'], df['prediction'])
# applied to whole of dataset-
# accuracy = 68%
# AUC = 73...
# could remake plots with these numbers.

accuracies = []
aucs = []
for s in xrange(2006,2017):
    temp = df.loc[df['season'] == s]
    accuracies.append(metrics.accuracy_score(temp['winner'], temp['prediction_class']))
    aucs.append(metrics.roc_auc_score(temp['winner'], temp['prediction']))

forPlot = pd.DataFrame(
    {'season': xrange(2006,2017),
     'accuracy': accuracies,
     'auc': aucs
    })

#plot this for blog post
pdf = PdfPages("{}/nba_model/docs/by_season.pdf".format(os.path.expanduser("~")))
plot1, ax =plt.subplots(figsize=(7.5,5))
plt.plot(forPlot['season'],forPlot['accuracy'],marker='o', linestyle='-', color='b')
plt.ylabel('Model Accuracy')
plt.xlabel('NBA Season')
plt.title('Model Accuracy by NBA Season')

pdf.savefig()
plt.close()

plot2, ax =plt.subplots(figsize=(7.5,5))
plt.plot(forPlot['season'],forPlot['auc'],marker='o', linestyle='-', color='b')
plt.ylabel('Model AUC')
plt.xlabel('NBA Season')
plt.title('Model AUC by NBA Season')
pdf.savefig()
plt.close()

pdf.close()


# investigate a bit

for s in xrange(2006,2017):
    temp = df.loc[df['season'] == s]
    print s, temp['winner'].groupby(df['east_west']).mean()

for s in xrange(2006,2017):
    temp = df.loc[df['season'] == s]
    print s, temp['winner'].groupby(df['daysoff_diff_bucket']).mean()

for s in xrange(2006,2017):
    temp = df.loc[df['season'] == s]
    print s, temp['winner'].groupby(df['daysoff_home_bucket']).mean()



