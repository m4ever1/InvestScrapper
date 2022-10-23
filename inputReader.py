import pandas as pd
from functools import cmp_to_key

def getInput(filename = ""):
    if not filename:
        filename = "./cpp/bin/output.txt"
    f = open(filename)
    return f

def parseInput(filename = ""):
    file = getInput(filename)
    entries = []
    for line in file.readlines():
        entry = {}
        tokens = line.split("|")
        # first token is tickers ex: ABV,XPT,APPL,...,...,EDP
        entry["tickers"] = list(tokens[0].split(","))
        entry["length"] = len(entry["tickers"])
        entry["return"] = float(tokens[1])
        entry["diversity"] = float(tokens[2])
        entries.append(entry)
    file.close()
    return entries

def buildDataFrame(filename = ""):
    df = pd.DataFrame.from_dict(parseInput(filename))
    return df

def optimalTickers(filename = ""):
    entries = parseInput(filename)

def compareEntries(lhs,rhs):
    if lhs["length"] == rhs["length"]:
        if lhs["return"] == rhs["return"]:
            if lhs["diversity"] < rhs["diversity"]:
                return -1
            else:
                return 1
        elif lhs["return"] < rhs["return"]:
            return -1
        else:
            return 1
    elif lhs["length"] < rhs["length"]:
        return -1
    else:
        return 1 

def buildPortfolio(entries: list):
    compareKey = cmp_to_key(compareEntries)
    return sorted(entries, key=compareKey, reverse=True)[0]["tickers"]
    