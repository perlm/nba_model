setwd("/home/jason/")
library(ggplot2)
df = read.table("temp.csv", header=TRUE, sep=",", quote='', nrows=250000)

ggplot(df,aes(x=prediction,y=overtime)) +
xlab('Predicted Home Win Probability') + 
ylab('Overtime Frequency')+ ggtitle('Relationship between NBA win probability and OT') +
xlim(c(0,1))+#ylim(c(0,0.25))+
stat_smooth(method='loess',se=TRUE)
