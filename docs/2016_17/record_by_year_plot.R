setwd("/home/jason/nba_model/raw")
library(ggplot2)
df06 = read.table("games_2007.csv", header=TRUE, sep=",", quote='', nrows=250000)
df07 = read.table("games_2008.csv", header=TRUE, sep=",", quote='', nrows=250000)
df08 = read.table("games_2009.csv", header=TRUE, sep=",", quote='', nrows=250000)
df09 = read.table("games_2010.csv", header=TRUE, sep=",", quote='', nrows=250000)
df10 = read.table("games_2011.csv", header=TRUE, sep=",", quote='', nrows=250000)
df11 = read.table("games_2012.csv", header=TRUE, sep=",", quote='', nrows=250000)
df12 = read.table("games_2013.csv", header=TRUE, sep=",", quote='', nrows=250000)
df13 = read.table("games_2014.csv", header=TRUE, sep=",", quote='', nrows=250000)
df14 = read.table("games_2015.csv", header=TRUE, sep=",", quote='', nrows=250000)
df15 = read.table("games_2016.csv", header=TRUE, sep=",", quote='', nrows=250000)
df16 = read.table("games_2017.csv", header=TRUE, sep=",", quote='', nrows=250000)

df06$season <- 2006
df07$season <- 2007
df08$season <- 2008
df09$season <- 2009
df10$season <- 2010
df11$season <- 2011
df12$season <- 2012
df13$season <- 2013
df14$season <- 2014
df15$season <- 2015
df16$season <- 2016


df <- rbind(rbind(rbind(rbind(rbind(rbind(rbind(rbind(rbind(rbind(df06,df07),df08),df09),df10),df11),df12),df13),df14),df15),df16)

df$win_round <- round(df$winrate_home,1)

forPlot1 <- as.data.frame(xtabs(df,formula=winner~win_round)/xtabs(df,formula=~win_round))
forPlot <- as.data.frame(xtabs(df,formula=winner~win_round+season)/xtabs(df,formula=~win_round+season))

ggplot() +
  geom_point(data=forPlot,aes(x=win_round,y=Freq,group=season,color=season),size=2,alpha=0.1)+
  geom_line(data=subset(forPlot,season!=2016),aes(x=win_round,y=Freq,group=season,color=season),size=2,alpha=0.1)+
  geom_line(data=subset(forPlot,season==2016),aes(x=win_round,y=Freq,group=season,color=season),size=2,alpha=0.9)+
  geom_line(data=forPlot1,aes(x=win_round,y=Freq,group=1),size=2,alpha=0.5)+
  xlab('Home Team Record') + 
  ylab('Home Team Win Probability') +
  theme(panel.background = element_blank())


ggplot() +
  xlab('Home Team Record') + 
  ylab('Home Team Win Probability') +
  theme(panel.background = element_blank())+
  stat_smooth(data=df,aes(x=winrate_home,y=winner,group=season,color=season),method='loess',se=FALSE,alpha=0.25,size=0.5)+
  stat_smooth(data=subset(df,season==2016),aes(x=winrate_home,y=winner),color="Blue",method='loess',se=FALSE,alpha=0.5,size=2)+
  stat_smooth(data=subset(df,season!=2016),aes(x=winrate_home,y=winner),color="Green",method='loess',se=TRUE,alpha=0.5,size=2)+
  ggtitle('NBA Win Probability by Record')



ggplot(data=df,aes(x=winrate_home,y=winner,color=as.factor(season),group=season)) +
  xlab('Home Team Record') + 
  ylab('Home Team Win Probability') +
  theme(panel.background = element_blank())+
  stat_smooth(method='loess',se=FALSE,alpha=0.5,size=2)


df$win_pos <- ifelse(df$winrate_home>0.5,1,0)
forPlot <- as.data.frame(xtabs(df,formula=winner~win_pos+season)/xtabs(df,formula=~win_pos+season))

ggplot(data=forPlot,aes(x=season,y=Freq,color=win_pos,group=win_pos))+
  geom_point(size=3,alpha=0.5)+geom_line(size=3,alpha=0.5)+
  #theme(panel.background = element_blank())+ 
  scale_color_discrete(name='Winning Home Team') + 
  xlab('Season')+ylab('Win Rate')+
  ggtitle('Win Rate by Record')


