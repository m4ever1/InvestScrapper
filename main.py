import investpy
import requests
from investpy.utils.extra import random_user_agent
import bs4
import pandas
import json

df = investpy.get_funds_dict(country="Portugal")

head = {
    "User-Agent": random_user_agent(),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "text/html",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
f = open("fundDict.json", "w")   
fundsInfoDict = {}
# get tables
for fund in df:
    print(fund["tag"])
    tag = fund["tag"]    
    url = "https://www.investing.com/funds/" + tag + "-holdings"
    req = requests.get(url, headers=head)
    soup = bs4.BeautifulSoup(req.text, "html.parser")

    tables = soup.find_all("table", attrs={"class": "genTbl"})
    print(len(tables))
    getTable = [False, False, False, False]
    if soup.find("div", attrs={"class":"asset section"}):
        getTable[0] = True
    if soup.find("div", attrs={"class" : "syleBox section"})
        getTable[1] = True
    if soup.find("div") #h3 Sector Allocation
    dfs = pandas.read_html(str(tables[0:4]))
    print(dfs)
    for df in dfs:
        csvF = open("./Funds/" + fund["name"] + ".csv", "a")
        df.to_csv("./Funds/" + fund["name"] + ".csv", mode="a")
        csvF.write("\n")
        csvF.close()

    fundsInfoDict[fund["name"]] = df.to_dict()

json.dump(fundsInfoDict, f, sort_keys=True, indent=4)
f.close()
# styleBox = soup.find("table",attrs={"class": "boxBoard"})
