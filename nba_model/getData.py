from bs4 import BeautifulSoup
import requests, sys, time, re,datetime, os
import numpy as np
import pandas as pd

#####
# The objective of this script is to scrape NBA data so I can make a predictive model!
#####

def getSoup(url):
	# scraping
	while True:
		try:
			r = requests.get(url)
			return BeautifulSoup(r.text, 'html.parser')
		except requests.ConnectionError:time.sleep(1)

def getGames(year,predicting=False):
    #####
    # process for scraping and manipulating data to prep for modeling!
    # return as dataframe!
    #####
    
    #########
    ### SCRAPE!
    # 0 <- for 2015-16 season, is '2016', 1<- lowercase month
    months = ['october','november','december','january','february','march','april']
    baseurl = 'http://www.basketball-reference.com/leagues/NBA_{0}_games-{1}.html'

    # get all games from monthly list
    df = None
    for m in months:
        r = getSoup(baseurl.format(year,m))
        d = parse_games(r)
        
        ######################3
        # remove games not yet played
        if year==2017 and not predicting:
            d = d.loc[(d['home_points']!=0)]

        if df is None:
            df = d
        else:
            df = df.append(d,ignore_index=True)
    
    if year not in (2012,2017):
	    assert (len(df)>(82*30./2)), 'Not all games found for year={0}!'.format(year)
    
    ####################
    # Manipulate!
    # create some cumulative/inter-game stats!
    df = create_features(df)
    
    #####################
    # Prune for modeling!
    # remove games after 82. remove games before 10 for modeling.
    df = df.loc[(df['game_count_home']>10) & (df['game_count_home'] <=82) & (df['game_count_away']>10) & (df['game_count_away'] <=82)]
    
    if not predicting:
        return df
    else:
        return df.loc[(df['date']==datetime.date.today())]
    
def create_features(df):
    #########################
    # create some cumulative/inter-game stats!
    ##########################
    
    # winning percentage, previous 1,3,5,10 games, win streak, cumulative point differential
    # previous meeting of these teams
    # number of games played this season, days between games, distance travelled,
    

    # assuming it's already in chronological order!
    # remember it's tricky, since I want info on both the home and away teams!
    # define some variables from the perspective of each team. then join them back to df.
    # I want the features to describe the team heading into that game, rather than after that game.
    teams = df.home_team.unique().tolist()
    assert (len(teams)==30), 'wrong number of teams!'
    
    
    for iii, t in enumerate(teams):
        #print t
        a = df.loc[(df['away_team'] == t) | (df['home_team'] == t)].copy(deep=True)
        
        # cumulative games
        a['game_count'] = range(1,len(a)+1)
        
        # days between games
        a['daysoff']    = (a['date'] - a['date'].shift(1))/np.timedelta64(1, 'D')-1
        
        # point differential
        a.loc[(a['away_team']==t),'pointdiff'] =  a['away_points']-a['home_points']
        a.loc[(a['home_team']==t),'pointdiff'] =  a['home_points']-a['away_points']
    
        # cum point differential/game
        #a['cumpointdiffAfter'] = pd.expanding_mean(a['pointdiff'])
        a['cumpointdiffAfter'] = a['pointdiff'].expanding(min_periods=1).mean()
        a['cumpointdiff'] = a['cumpointdiffAfter'].shift(1)
        
        # win (whether this team won, regardless of whether they were home or away)
        a['win']    = 0
        a.loc[(a['pointdiff']>0),'win'] = 1
        
        # cumulative wins/game
        #a['winrateAfter'] = pd.expanding_mean(a['win'])
        a['winrateAfter'] = a['win'].expanding(min_periods=1).mean()
        a['winrate'] = a['winrateAfter'].shift(1)
        
        # prior game is OT
        a['afterOT']  = a['overtime'].shift(1)/1.0
        
        # cumulative wins/game recent
        a['win1']  = a['win'].shift(1)/1.0
        a['win3']  = (a['win'].shift(1)+a['win'].shift(2)+a['win'].shift(3))/3.0
        a['win5']  = (a['win'].shift(1)+a['win'].shift(2)+a['win'].shift(3)+a['win'].shift(4)+a['win'].shift(5))/5.0
        a['win10'] = (a['win'].shift(1)+a['win'].shift(2)+a['win'].shift(3)+a['win'].shift(4)+a['win'].shift(5)+a['win'].shift(6)+a['win'].shift(7)+a['win'].shift(8)+a['win'].shift(9)+a['win'].shift(10))/10.0
    
        ## Now join back to df! - use original indexing!
        b = a.loc[(a['away_team'] == t)]
        b = b[['game_count','daysoff','cumpointdiff','winrate','win1','win3','win5','win10','afterOT']]
        b.columns = ['game_count_away','daysoff_away','cumpointdiff_away','winrate_away','win1_away','win3_away','win5_away','win10_away','afterOT_away']
        c = a.loc[(a['home_team'] == t)]
        c = c[['game_count','daysoff','cumpointdiff','winrate','win1','win3','win5','win10','afterOT']]
        c.columns = ['game_count_home','daysoff_home','cumpointdiff_home','winrate_home','win1_home','win3_home','win5_home','win10_home','afterOT_home']
		
	# to avoid creating duplicate column names
	if iii==0:
		df = pd.merge(df,b,how='left',left_index=True,right_index=True)
		df = pd.merge(df,c,how='left',left_index=True,right_index=True)
	else:
		df = df.combine_first(b)
		df = df.combine_first(c)
	
    return df


def parse_games(r):
    ################
    # take in html and return dataframe with game info on website
    ################
    
    #remove section with playoffs - not getting it with bs or re. 
    #tr class="thead"><th colspan="9">Playoffs</th></tr>
    #playoffcheck = re.search(r'(.*)tr class="thead"><th colspan="9">Playoffs</th></tr>', str(r), re.M|re.I)

    dates = []
    for entry in r.find_all("th", { "data-stat":"date_game" }):
        if entry.text == 'Date':continue
        dates.append(entry.text)

    times = []
    for entry in r.find_all("td", { "data-stat":"game_start_time" }):
        times.append(entry.text)

    away_teams = []
    for entry in r.find_all("td", { "data-stat":"visitor_team_name" }):
        away_teams.append(entry.text)

    away_points = []
    for entry in r.find_all("td", { "data-stat":"visitor_pts" }):
        if entry.text=='':
            away_points.append(int(0))
        else:
            away_points.append(int(entry.text))

    home_teams = []
    for entry in r.find_all("td", { "data-stat":"home_team_name" }):
        home_teams.append(entry.text)

    home_points = []
    for entry in r.find_all("td", { "data-stat":"home_pts" }):
        if entry.text=='':
            home_points.append(int(0))
        else:
            home_points.append(int(entry.text))

    ots = []
    for entry in r.find_all("td", { "data-stat":"overtimes" }):
        if entry.text=='':
            ots.append(int(0))
        else:
            ots.append(int(1.0))

    assert (len(set([len(dates),len(times),len(away_teams),len(away_points),len(home_teams),len(home_points),len(ots)])) == 1), 'varying number of game inputs!'

    df = pd.DataFrame(
        {'date': dates,
         'time': times,
         'away_team': away_teams,
         'away_points': away_points,
         'home_team': home_teams,
         'home_points': home_points,
         'overtime' : ots
    })
    
    
    df['winner'] = np.where(df['home_points']>df['away_points'], 1, 0)
    df['date'] = pd.to_datetime(df['date'], format='%a, %b %d, %Y')

    return df


def get_all_data_for_modeling():
    ####################
    # get data for modeling!
    # scrapes, manipulated df, and then writes to csv.
    #################
    
    for year in xrange(2007,2018):
        df = getGames(year)
        df.to_csv('{0}/nba_model/data/games_{1}.csv'.format(os.path.expanduser("~"),year), index=False)

def get_new_data_for_modeling():
    ####################
    # get data for modeling!
    # scrapes, manipulated df, and then writes to csv.
    # only need 2017 season, since other season's aren't changing
    #################
    
    year=2017
    df = getGames(year)
    df.to_csv('{0}/nba_model/data/games_{1}.csv'.format(os.path.expanduser("~"),year), index=False)

def get_data_for_predicting():
    df = getGames(2017,predicting=True)
    df.to_csv('{0}/nba_model/data/predict_games.csv'.format(os.path.expanduser("~")), index=False)

if __name__ == '__main__':
    get_all_data_for_modeling()
    get_new_data_for_modeling()
    get_data_for_predicting()

