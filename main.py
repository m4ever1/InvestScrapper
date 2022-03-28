import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
import numpy as np

# store = pd.HDFStore('stocks.h5')

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
sectors_dict = {}
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    sector = row.findAll('td')[3].text
    ticker = ticker.strip()
    tickers.append(ticker)
    sectors_dict[ticker] = sector

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
data.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'))
data.rename(columns = {"Unnamed: 0_level_0" : "Stat", "Unnamed: 0_level_1" : "Ticker"})
df = data["Adj Close"]
df = df.dropna(how="all")
df = df.dropna(axis=1)
# df = df.interpolate()
G = pd.DataFrame()


# Iterate through every ticker
G = np.log(df).shift(-1) - np.log(df)

G.dropna(how='all')

# C = np.mean(G.multiply)

rows = []
for ticker_i in G:
    column = []
    for ticker_j in G:
        column.append((np.mean(G[ticker_i].multiply(G[ticker_j])) - np.mean(G[ticker_i])*np.mean(G[ticker_j]))/(G[ticker_i].std()*G[ticker_j].std()))
    rows.append(column)

C = pd.DataFrame(rows, columns=G.columns, index=G.columns)


print(data)

