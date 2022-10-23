import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import  pylab as pl
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def getSectorsDict(data, num_clusters = 6):
    #Loading the data
    # data = pd.read_csv("stocks.csv")
    # data = data.set_index("Date")

    # data = data.dropna(how="all")
    # data = data.dropna(axis=1)
    # data = data.interpolate()
    # data = data.sort_index()
    
    #Outlier elimination
    # data = data[data.columns[(np.abs(stats.zscore(data,axis=0)) < 3).all(axis=0)]]

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
    # kmeans = KMeans(n_clusters = 11).fit(X)
    # centroids = kmeans.cluster_centers_
    # # pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    # # pl.show()


    # # print(returns.idxmax())
    # tickerToDrop = returns.idxmax() if returns[returns.idxmax()] > returns[returns.idxmin()] else returns.idxmin() 
    # ret_var.drop(tickerToDrop, inplace =True)


    X = ret_var.values
    kmeans =KMeans(num_clusters, random_state=3425).fit(X)
    centroids = kmeans.cluster_centers_
    # pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
    # pl.show()


    ticker = pd.DataFrame({'ticker' : ret_var.index})
    cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    zip_iterator = zip(ret_var.index, kmeans.labels_)
    dataOut = dict(zip_iterator)
    # fullData = pd.concat([ticker, cluster_labels],axis = 1)
    # fullData = fullData.set_index('ticker')
    
    return dataOut

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

            fig = pl.figure(num=1, clear=True)
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
            fig.savefig(f"images/plot-{n_clusters}-clusters.png", dpi=400)
            fig.clear()
            # pl.close(fig)
            fig = pl.figure(num=1, clear=True)
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
            fig.savefig(f"images/line-{n_clusters}-clusters.png", dpi=400)
            # pl.close(fig)



    pl.close()
    
def getClusterGraph(n_clusters: int, labels, X):
    fig = pl.figure(num=1, clear=True)
    ax2 = pl.axes()
    fig.set_size_inches(20, 2)
    
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.set_yticks([])
    ax2.vlines(
        X, ymax=n_clusters*1.2,ymin=-n_clusters*0.8, colors=colors,zorder=-1
    )

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Stock variance")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    # pl.suptitle(
    #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
    #     % n_clusters,
    #     fontsize=14,
    #     fontweight="bold",
    # )
    fig.savefig(f"images/line-{n_clusters}-clusters-GICS.png", dpi=400)
    # pl.close(fig)