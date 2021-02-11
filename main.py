import bs4 as bs
import requests
import yfinance as yf
import datetime


resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]
start = datetime.datetime(2005, 1, 1)
end = datetime.datetime(2021, 1, 1)
data = yf.download(tickers, start=start, end=end)
data.to_csv('stocks.csv')
print(data)
