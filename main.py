import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd

# store = pd.HDFStore('stocks.h5')

# resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# soup = bs.BeautifulSoup(resp.text, 'lxml')
# table = soup.find('table', {'class': 'wikitable sortable'})
# tickers = []
# for row in table.findAll('tr')[1:]:
#     ticker = row.findAll('td')[0].text
#     tickers.append(ticker)

# tickers = [s.replace('\n', '') for s in tickers]
# start = datetime.datetime(2017, 1, 1)
# end = datetime.datetime(2021, 1, 1)
# data = yf.download(tickers, start=start, end=end)

# data.to_csv("stocks.csv")
# store["data"] = data

# data = store['data']
data = pd.read_csv("stocks.csv", header=[0, 1])
data.columns = pd.MultiIndex.from_tuples(data.columns)

new_header = data.iloc[0] #grab the first row for the header
data = data[1:] #take the data less the header row
data.set_index(data.columns[0])

print(data)


