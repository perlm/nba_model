import requests, pickle, bz2,datetime, boto3
#from bs4 import BeautifulSoup

# for aws lambda:
#   sudo pip install requests datetime boto3 bs4 pip -t ~/nba_model/aws/^C
# aws is in utc timezone. can't pickle beautiful soup objects.

print('Loading function')


def lambda_handler(event, context):
	getTodaysGamesAWS(2018)
	#return unparsed
        #return BeautifulSoup(unparsed.text, 'html.parser')
	return True



def getSoup(url):
	while True:
		try:
			r = requests.get(url)
			return r
			#return BeautifulSoup(r.text, 'html.parser')
		except requests.ConnectionError:time.sleep(1)

def getTodaysGamesAWS(year):
	baseurl = 'http://www.basketball-reference.com/leagues/NBA_{0}_games-{1}.html'
	monthDict = dict([(10,'october'),(11,'november'),(12,'december'),(1,'january'),(2,'february'),(3,'march'),(4,'april')])
	monthList = [10,11,12,1,2,3,4]
	t = datetime.date.today()
	now = t.month
	if now in monthList:
		for mon in monthList:
			r = getSoup(baseurl.format(year,monthDict[mon]))

			with bz2.BZ2File("/tmp/nba_unparsed_{0}.pickle".format(mon),"w") as f:pickle.dump(r, f)

			s3 = boto3.resource('s3')
			BUCKET_NAME = 'jeopardydata'
			s3.meta.client.upload_file('/tmp/nba_unparsed_{0}.pickle'.format(mon), BUCKET_NAME, 'nba_unparsed_{0}.pickle'.format(mon))

			if mon == now:
				break


