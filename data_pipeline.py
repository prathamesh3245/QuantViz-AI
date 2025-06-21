import yfinance as yf


btc = yf.Ticker("BTC-USD")
btc_data = btc.history(period='1d', interval='5m')
btc_data.to_csv("data/live_btc.csv")

import pandas as pd
import requests, re, pathlib, html

txt = requests.get("https://www.sec.gov/litigation/litreleases.shtml", headers={"User-Agent": "Mozilla/5.0"}).text
pattern = re.compile(r'(<time[^>]+>([^<]+)</time>).*?href="(/enforcement-litigation/litigation-release[^"]+)".*?>(LR‑\d+)', re.S)
rows = [(html.unescape(d), html.unescape(r), n, f'https://www.sec.gov{u}')
        for d,r,u,n in pattern.findall(txt)]
pd.DataFrame(rows, columns=["release_date","respondents","release_no","url"])\
  .to_csv("data/sec_insider_cases.csv", index=False)


from transformers import pipeline  

ner = pipeline("ner", model="dslim/bert-base-NER")  
sec_entities = ner(sec_cases["text"].to_string())