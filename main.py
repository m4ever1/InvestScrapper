import investpy
import requests
from investpy.utils.extra import random_user_agent
import bs4
import pandas

df = investpy.get_funds_dict(country="Portugal")

head = {
    "User-Agent": random_user_agent(),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "text/html",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# for fund in df:
fund = df[2]
print(fund["tag"])
tag = fund["tag"]    
url = "https://www.investing.com/funds/" + tag + "-holdings"
req = requests.get(url, headers=head)
soup = bs4.BeautifulSoup(req.text, "html.parser")
# print(head["User-Agent"])
tables = soup.find_all("table", attrs={"class": "genTbl"})
dfs = []
for table in tables:
    dfs.append(pandas.read_html(str(table)))
print(dfs)