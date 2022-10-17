# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('/images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg, '\n'
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi="2400")
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi="2400")



    pl.close()

# %%
import datetime, requests, re, sys, os
from multiprocessing.sharedctypes import Value
from logging import exception
from email.policy import default
import bs4 as bs
import yfinance as yf
from collections import defaultdict
import pandas as pd
import numpy as np
from kmeans import getSectorsDict
from math import sqrt
import  pylab as pl
from collections import OrderedDict

def getFileDir():
    if getattr(sys, 'frozen', False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)

    return datadir

def getTable():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    return table

def compareSectors(df: pd.DataFrame):
    gicsSects = getGICSSectors(df)
    kmeansSects = getSectorsDict(df)[0]
    countDict = {}

    print("GICS SECTORS")

    # for gicsSect in gicsSects.values():
    #     tickerList = gicsSect
    #     for ticker in tickerList:
    #         try:
    #             if kmeansSects[ticker] in countDict:
    #                 countDict[kmeansSects[ticker]] += 1
    #             else:
    #                 countDict[kmeansSects[ticker]] = 1
    #         except:
    #             pass

    # return countDict

# reads tickers from a text file in a comma separated list e.g. (A,AAP,APPL,AAL) 
# DOES NOT SANITIZE: SPACES IN BETWEEN, BRACKETS OR COMMAS, it only eliminates duplicates and sorts the tickers lexicographycally
def readTickersFromFile(fileName: str = None) -> list:
    if fileName is None:
        fileName = "nasdaqTickers2013-2014.txt" 
    with open(fileName, "r") as file:
        text = file.read()
        tickers = text.split(',')
        tickers = sorted(set(tickers)) # set() filters non-uniques, and sorted... sorts.

    return tickers

def scrapeTickerList(index: str = "spy"):
    url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tickerColumn = 0
    if index != "spy":
        if index == "nasdaq":
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tickerColumn = 1
            
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable', 'id' : 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[tickerColumn].text
        ticker = ticker.strip()
        tickers.append(ticker)

    tickers = [s.replace('\n', '') for s in tickers]

    return tickers

def downloadStockData(dateStart: dict, dateEnd: dict, dataSet: str = "spy") -> pd.DataFrame:
    if dataSet == "spy":
        tickerList = scrapeTickerList("spy")
    elif dataSet == "nasdaq":
        if dateEnd["year"] < 2015:
            tickerList = readTickersFromFile()
        else:
            tickerList = scrapeTickerList("nasdaq")
    else: 
        raise Exception("Invalid data set")

    start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
    end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
    
    data = yf.download(tickerList, start=start, end=end)
    data["Adj Close"].to_csv("stocks.csv")
    data = data["Adj Close"]
    data = data.dropna(how="all")
    data = data.dropna(axis=1)
    data = data.interpolate()
    
    return data

def readDataFromCsv() -> pd.DataFrame:
    df = pd.read_csv("stocks.csv")
    df = df.set_index("Date")
    df = df.sort_index()
    
    return df
    # df = df.dropna(how="all")
    # df = df.dropna(axis=1)
    # df = df.interpolate()
    # df = df.sort_index()

def filterStocksByDate(df: pd.DataFrame, dateStart: dict, dateEnd: dict) -> pd.DataFrame:
    start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
    end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
    
    dfOut = df.loc[str(start):str(end)]
    dfOut = dfOut.dropna(how="all")
    dfOut = dfOut.dropna(axis=1)
    dfOut = dfOut.interpolate()
    return dfOut

def convertToInputFile(df: pd.DataFrame, dateStart: dict, dateEnd: dict) -> str:
    G = pd.DataFrame()
    G = np.log(df).shift(-1) - np.log(df)

    G = G.dropna(how='all')

    # rows = []
    # for ticker_i in G:
    #     column = []
    #     for ticker_j in G:
    #         column.append((np.mean(G[ticker_i].multiply(G[ticker_j])) - np.mean(G[ticker_i])*np.mean(G[ticker_j]))/(G[ticker_i].std()*G[ticker_j].std()))
    #     rows.append(column)

    # C = pd.DataFrame(rows, columns=G.columns, index=G.columns)

    # t = G.shape[0]
    # n = G.shape[1]

    # Q = t/n

    # lMax = 1 + 1/Q + 2*np.sqrt(1/Q)
    # lMin = 1 + 1/Q - 2*np.sqrt(1/Q)

    pctChangeDf = df.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

    # stocksToTransact = []
    # for stck in list(sectors_dict.values()):
    #     stock_name = stck
    #     stocksToTransact += stock_name 

    transactions = pctChangeDf
    # stockToSector = {stock: index for index, tuple in enumerate(sectors_dict.items()) for stock in tuple[1]}
    stockToSector, droppedTicker = getSectorsDict(df)
    # listOfTransLists = '\r\n'.join(f"{' '.join(map(str, range(1, len(transactions.columns) + 1)))}:{transactions.iloc[i].sum()}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
    transactions = transactions.drop(droppedTicker, axis=1, errors="ignore")

    listOfTransLists = '\r\n'.join(f"{' '.join(map(str, transactions.columns))}:{' '.join(map(str, [stockToSector[index] for index in transactions.iloc[i].index]))}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
    listOfTransLists = re.sub("  +", " ", listOfTransLists)
    
    dirname = getFileDir()

    outputFile = f'input-{dateStart["year"]}-{dateStart["month"]}-{dateStart["day"]}-To-{dateEnd["year"]}-{dateEnd["month"]}-{dateEnd["day"]}-.txt'
    filePath = os.path.join(dirname, "")
    absolutePath = os.path.join(filePath, outputFile)

    with open(absolutePath, "w") as f:
        f.write(listOfTransLists)
        print(f"Saved stocks to file: {absolutePath}")
    
    return absolutePath

def getGICSSectors(df: pd.DataFrame) -> defaultdict(list):
    # sectors_dict = defaultdict(list)
    sectors_dict = {}
    table = getTable()
    for ticker in df.columns:
        try:
            row = table.find("a", string=ticker).parent.parent
            sector = row.findAll('td')[3].text
            sectors_dict[ticker] = sector
        except:
            pass

    sectorList = []
    tickersToDrop = []

    for ticker in df.columns:
        try:
            sectorList.append(sectors_dict[ticker])
        except:
            tickersToDrop.append(ticker)

    sectorListNr = [{val: key for key, val in enumerate(
        OrderedDict.fromkeys(sectorList))}
        [ele] for ele in sectorList]

    returns = df.drop(tickersToDrop, axis=1).pct_change().mean() * 252
    variance = df.drop(tickersToDrop, axis=1).pct_change().std() * sqrt(252)
    returns.columns = ["Returns"]

    variance.columns = ["Variance"]
    sectors = pd.DataFrame(sectorListNr, index=returns.index)

    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns, variance, sectors], axis = 1).dropna()
    ret_var.columns = ["Returns","Variance", "Sector"]

    X =  ret_var.values #Converting ret_var into nummpy array
    # X = ((df-df.mean())/(df.std())).values #Converting ret_var into nummpy array
    
    pl.scatter(X[:,0],X[:,1], c = X[:,2], cmap ="rainbow")
    pl.show()
    
    # ticker = pd.DataFrame({'ticker' : ret_var.index})
    # cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    # zip_iterator = zip(ret_var.index, kmeans.labels_)
    # dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    # return dataOut

def generateInputFile(dateStart: dict, dateEnd: dict, granularity: str, dataSet: str = "spy") -> list:
    assert(dateStart["day"] == dateEnd["day"])
    if granularity == "yearly":
        assert(dateStart["month"] == dateStart["month"])
    
    listOfFiles = []
    df = downloadStockData(dateStart, dateEnd, dataSet)

    auxDateStart = dateStart.copy()
    while(auxDateStart != dateEnd):
        if auxDateStart["month"] == 12:
            auxDateStart["year"] += 1
            auxDateStart["month"] = 1

        auxDateEnd = auxDateStart.copy()
        if granularity == "trimester" :
            if auxDateEnd["month"] + 3 > 12:
                auxDateEnd["month"] = (auxDateEnd["month"]+3)%12 if (auxDateEnd["month"]+3)%12 != 0 else 1
                auxDateEnd["year"] += 1
            else:
                auxDateEnd["month"] += 3
        else:
            auxDateEnd[granularity] += 1
        
        auxDf = filterStocksByDate(df, auxDateStart, auxDateEnd)
        fname = convertToInputFile(auxDf, auxDateStart, auxDateEnd)
        
        if granularity == "trimester":
            if auxDateStart["month"] + 3 > 12:
                auxDateStart["month"] = (auxDateStart["month"]+3)%12 if (auxDateStart["month"]+3)%12 != 0 else 1
                auxDateStart["year"] += 1
            else:
                auxDateStart["month"] += 3
        else:
            auxDateStart[granularity] += 1

        listOfFiles.append(fname)
        
    return listOfFiles

# %%
import pandas as pd
import DataMining
import Utils
# import weightCalculate

dateStart = {
        "year" : 2012,
        "month" : 8,
        "day"   : 1
    }
dateEnd = {
        "year" : 2013,
            "month" : 8,
            "day"   : 1
        }

# listOfFiles = Utils.generateInputFile(dateStart, dateEnd, 'yearly', "nasdaq")


# DataMining.mine(listOfFiles)

# dateStart = {
#         "year" : 2006,
#         "month" : 9,
#         "day"   : 1
#     }
# dateEnd = {
#         "year" : 2022,
#             "month" : 9,
#             "day"   : 1
#         }

# listOfFiles = Utils.generateInputFile(dateStart, dateEnd, 'month') 

# DataMining.mine(listOfFiles)
#store = pd.HDFStore('stocks.h5')



# dirname = os.path.dirname(__file__)


# weights = weightCalculate.getWeights(df)

# %%
data = downloadStockData(dateStart, dateEnd, 'nasdaq')

# %%
data

# %%
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array

# %%
silhouetteAnalasys(X)

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg, '\n'
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi="2400")
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi="2400")



    pl.close()

# %%
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array

# %%
silhouetteAnalasys(X)

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                "For n_clusters =" +
                n_clusters +
                "The average silhouette_score is :" +
                silhouette_avg + '\n'
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi="2400")
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi="2400")



    pl.close()

# %%
silhouetteAnalasys(X)

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters}The average silhouette_score is :{silhouette_avg} + \n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi="2400")
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi="2400")



    pl.close()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg} + \n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi="2400")
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi="2400")



    pl.close()

# %%
silhouetteAnalasys(X)

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg} + \n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi=2400)
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi=2400)



    pl.close()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg} + \n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi=2400)
            pl.close(fig)
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi=2400)
            pl.close(fig)



    pl.close()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg}\n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi=2400)
            pl.close(fig)
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi=2400)
            pl.close(fig)



    pl.close()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg}\n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi=2400)
            pl.close(fig)
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi=2400)
            pl.close(fig)



    pl.close()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg}\n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi=2400)
            fig.clear()
            pl.close(fig)
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi=2400)
            fig.clear()
            pl.close(fig)



    pl.close()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging
from scipy import stats

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

    #Calculating annual mean returns and variances
    returns = data.pct_change().mean()/data.pct_change().var()
    returns.columns = ["Returns"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns], axis = 1).dropna()
    ret_var.columns = ["Returns"]


    X =  ret_var.values #Converting ret_var into nummpy array
    # sse = []
    # for k in range(2,15):
        
    #     kmeans = KMeans(n_clusters = k)
    #     kmeans.fit(X)
        
    #     sse.append(kmeans.inertia_) #SSE for each n_clusters
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

    #
    kmeans = KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 8).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    return dataOut, tickerToDrop

def reverseMapSectors(dfInput, dfOutput):
    pass

def silhouetteScore(X: pd.DataFrame):
    
    # fig, ax = pl.subplots(2, 2, figsize=(15,8))
    count = 1
    lastIter = 15
    firstIter = 2
    for k in range(firstIter,lastIter):
        # if not count%4:
        #     pl.show()
        #     fig, ax = pl.subplots(2, 2, figsize=(15,8))
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
        kmeans.fit(X)

        # q, mod = divmod(k, 2)
        # q = q%2
        score = silhouette_score(X, kmeans.labels_)

        # viz = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        # viz.fit(X)
        print('For %d clusters: Silhouetter Score: %.3f' %( k, score))
        # count += 1
    # pl.plot(range(2,15), sse)
    # pl.title("Elbow Curve")
    # pl.show()

def silhouetteAnalasys(X: pd.DataFrame):
    range_n_clusters = [2, 3, 4, 5, 6]

    with open('images/score.txt', 'w') as scoreFile:
        

        for n_clusters in range(2,20):
            # Create a subplot with 1 row and 2 columns

            fig = pl.figure()
            # fig2 = pl.figure()
            #  (ax1, ax2) = pl.subplots(2, 1)
            fig.set_size_inches(10, 10)

            ax1 = pl.axes()
            # ax2 = pl.axes()
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            scoreFile.write(
                f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg}\n"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette score")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi=1600)
            fig.clear()
            pl.close(fig)
            fig = pl.figure()
            ax2 = pl.axes()
            fig.set_size_inches(20, 2)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.set_yticks([])
            ax2.vlines(
                X[:, 0], ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                range(0, len(centers[:,0])),
                marker="o",
                c="white",
                alpha=1,
                s=500,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], i, marker="$%d$" % i, alpha=1, s=300, edgecolor="k", color='black')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Stock variance")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # pl.suptitle(
            #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            #     % n_clusters,
            #     fontsize=14,
            #     fontweight="bold",
            # )
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi=1600)
            fig.clear()
            pl.close(fig)



    pl.close()


