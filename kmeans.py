import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt
import  pylab as pl
import numpy as np

def getSectorsDict(data):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    #Calculating annual mean returns and variances
    returns = data.pct_change().mean() * 252
    variance = data.pct_change().std() * sqrt(252)
    returns.columns = ["Returns"]
    variance.columns = ["Variance"]
    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns, variance], axis = 1).dropna()
    ret_var.columns = ["Returns","Variance"]


    X =  ret_var.values #Converting ret_var into nummpy array
    sse = []
    for k in range(2,15):
        
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        
        sse.append(kmeans.inertia_) #SSE for each n_clusters
    pl.plot(range(2,15), sse)
    pl.title("Elbow Curve")
    pl.show()

    #
    kmeans = KMeans(n_clusters = 5).fit(X)
    centroids = kmeans.cluster_centers_
    pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    pl.show()


    print(returns.idxmax())
    tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(n_clusters = 11).fit(X)
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