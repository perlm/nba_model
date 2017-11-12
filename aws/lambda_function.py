import requests, pickle, bz2,datetime, boto3
#from bs4 import BeautifulSoup

# for aws lambda:
#   sudo pip install requests datetime boto3 bs4 pip -t ~/nba_model/aws/^C
# aws is in utc timezone. can't pickle beautiful soup objects.

print('Loading function')


def lambda_handler(event, context):
	BUCKET_NAME = 'jeopardydata'
	s3 = boto3.resource('s3')
	unparsed = getTodaysGamesAWS(2018)
	with bz2.BZ2File("/tmp/nba_unparsed.pickle","w") as f:pickle.dump(unparsed, f)
	s3.meta.client.upload_file('/tmp/nba_unparsed.pickle', BUCKET_NAME, 'nba_unparsed.pickle')
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
	month = dict([(10,'october'),(11,'november'),(12,'december'),(1,'january'),(2,'february'),(3,'march'),(4,'april')])
	t = datetime.date.today()
	m = month[t.month]
	r = getSoup(baseurl.format(year,m))
	return r


