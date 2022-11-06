###################################################################
# THIS IS DEAD CODE:                                              #
#                                                                 #
# prototype of code used for markovitz optimization,              #
# did not end up in final version.                                #
#                                                                 #
###################################################################

from inputReader import buildDataFrame
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
import numpy as np

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def getWeights(df):
    # outDf = buildDataFrame()

    # ticksOut = outDf.iloc[6].tickers

    normalizedDf = (df-df.mean())/df.std()

    retsIn= np.array(df.T)

    weights, rets, risks = optimal_portfolio(retsIn)

    return weights
