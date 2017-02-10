###
# code to make some plots
# used to investigate model performance
##

from ggplot import *
import pandas as pd

#plot_distribtuion(y_probs):
y = pd.DataFrame({'data':y_probs})
ggplot(y,aes(x='data')) + geom_histogram(binwidth=0.01,alpha=0.75)+xlab('Predicted Win Probability')+ggtitle('Distribution of Predictions in training set (2006-17)') + ylab('Count')

#plot_distribution_colorful(df,y_probs):
df['prediction'] = y_probs
df['days_off'] = pd.cut(df['daysoff_home'],bins=[-0.5,0.5,100],labels=['Back_to_Back','Rested'])

ggplot(df,aes(x='daysoff_home_bucket',y='prediction')) + \
geom_boxplot(size=100)+ \
xlab('Days Off') + ylab('Predicted Win Probability')+ggtitle('Distribution of Predictions in training set (2006-17)') + \
scale_y_continuous(limits=[0,1])


ggplot(df,aes(x='daysoff_home',y='prediction',color='winrate_home')) + \
geom_jitter(size=50,position = 'jitter')+ \
xlab('Days Off') + ylab('Predicted Win Probability')+ggtitle('Distribution of Predictions in training set (2006-17)') + \
scale_color_gradient(low = 'red', high = 'blue') + \
scale_y_continuous(limits=[0,1]) + scale_x_continuous(limits=[-0.5,8.5])


# stat2d, geom_tile don't really work. need to do this with ponits
df['winrate'] = pd.cut(df['winrate_home'],bins=[-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,100],labels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
forplot = df.groupby(['winrate','daysoff_home'])['prediction'].mean()
f = forplot.reset_index()

ggplot(f,aes(y='daysoff_home',x='winrate',color='prediction')) + \
geom_point(size=500) +\
ylab('Days Off') + xlab('Team Record')+ggtitle('Distribution of Predictions in training set (2006-16)') + \
scale_y_continuous(limits=[-0.25,2.25]) + scale_x_continuous(limits=[0,1.05]) +\
scale_color_gradient(low = 'red', high = 'blue')

# 
# python-ggplot won't stack or let me change factor order, so most of data is always hidden
#ggplot(df,aes(x='prediction',fill='days_off')) + \
#geom_histogram(binwidth=0.05,alpha=0.5)+ \
#xlab('Predicted Win Probability')+ggtitle('Distribution of Predictions in training set (2006-16)') + ylab('Count') + \
#scale_x_continuous(limits=[0,1])    


#plot_calibration
X['prediction'] = y_probs_lr
X['winner'] = Y
X = X.round({'prediction': 2})
forcalibration = X['winner'].groupby(X['prediction']).mean()
f = pd.DataFrame({'predict':forcalibration.index,'avgwin':forcalibration})
f['variance'] = abs(f['predict']-f['avgwin'])

f.to_csv('temp.csv', index=False)



ggplot(f,aes(x="predict",y="avgwin",color="variance")) + \
geom_point(size=100) + \
xlab('Predicted Win Probability') + \
ylab('Actual Win Frequency')+ggtitle('Calibration Analysis for NBA win model') + \
geom_abline(intercept=0,slope=1) + \
scale_y_continuous(limits=[0,1])+scale_x_continuous(limits=[0,1]) + \
scale_color_gradient(low = 'red', high = 'blue')


#plot_classification_by_month
df['prediction'] = y_probs
df['date'] = pd.to_datetime(df['date'])
# no october, since I'm excluding first 10 games.
months = ["Unknown",
      "January",
      "Febuary",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December"]
df['monthnumber'] = df['date'].dt.month
df['month']= df['monthnumber'].apply(lambda x: months[x])

# now group by month!
df['classified'] = 0
df.loc[(df['winner']==1) & (df['prediction']>0.5),'classified'] = 1
df.loc[(df['winner']==0) & (df['prediction']<0.5),'classified'] = 1

# now plot
forplot = df['classified'].groupby(df['month']).mean()
f = pd.DataFrame({'month':forplot.index,'avgcorrect':forplot})
ggplot(f,aes(x="month",y="avgcorrect"))+xlab('Month') + \
ylab('Classification Accuracy')+ggtitle('Classification Accuracy by month for NBA win model') + \
geom_point(size=100) 

#data doesn't imply change in accuracy over time.


#plot distance!
# not interesting!

forplot = df.round({'distance': -3})
forplot2 = forplot['winner'].groupby(forplot['distance']).mean()
forplot3 = forplot2.reset_index()

ggplot(forplot3,aes(x="distance",y="winner")) + \
geom_point(size=100) +\
xlab('Distance (km)') + \
ylab('Home Team Win Frequency')+ggtitle('Effect of travel distance on NBA win frequency') +\
scale_y_continuous(limits=[0.5,0.7]) 

+ stat_smooth(method='linear')


#################
# plot overtime!
#############33

df['prediction'] = y_probs_lr
forplot = df.round({'prediction': 2})
forplot2 = forplot['overtime'].groupby(forplot['prediction']).mean()
forplot3 = forplot2.reset_index()

ggplot(forplot3,aes(x="prediction",y="overtime")) + \
geom_point(size=100) +\
xlab('Predicted Home Win Probability') + \
ylab('Overtime Frequency')+ggtitle('Relationship between NBA win probability and OT') +\
scale_y_continuous(limits=[0.,0.2])+scale_x_continuous(limits=[0.,1.]) 


df['prediction'] = y_probs_lr
ggplot(df,aes(x="prediction",y="overtime")) + \
xlab('Predicted Home Win Probability') + \
ylab('Overtime Frequency')+ggtitle('Relationship between NBA win probability and OT') +\
scale_y_continuous(limits=[0.,0.2])+scale_x_continuous(limits=[0.,1.]) + \
stat_smooth(method='loess',se=False)

# try it in R
df[['prediction','overtime']].to_csv('/home/jason/temp.csv', index=False)









