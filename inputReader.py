import pandas as pd

def getInput(filename = ""):
    if not filename:
        filename = "./cpp/bin/output.txt"
    f = open(filename)
    return f

def parseInput():
    file = getInput()
    entries = []
    for line in file.readlines():
        entry = {}
        tokens = line.split("|")
        # first token is tickers ex: ABV,XPT,APPL,...,...,EDP
        entry["tickers"] = list(tokens[0].split(","))
        entry["return"] = float(tokens[1])
        entry["diversity"] = float(tokens[2])
        entries.append(entry)
    return entries

def buildDataFrame():
    df = pd.DataFrame.from_dict(parseInput())
    return df

def optimalTickers():
    entries = parseInput()
    