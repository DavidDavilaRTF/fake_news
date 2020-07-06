import pandas
import numpy
import sys
sys.path.append('C:\\web_driver\\')
from web_driver import *
import time
import requests
from lxml import html

start = time.time()
url = pandas.read_csv('C:\\fake_news\\doc\\url_gai.csv',engine = 'python',sep = ';')
url = url['FkN'].unique()
k = 1
nb_done = 0
for u in url:
    if k > nb_done:
        try:
            page = requests.get(u)
            page = html.fromstring(page.text)
            url_gaia = page.xpath('//div[@class = "td-module-thumb"]//a//@href')
            print((k,len(url)))
            url_gaia = pandas.DataFrame(url_gaia)
            url_gaia.columns = ['FkN']
            url_gaia.to_csv('C:\\fake_news\\doc\\url_gai_final.csv',sep = ';',index = False,mode = 'a',header = False)
        except:
            pass
    k += 1