import pandas
import numpy
import sys
sys.path.append('C:\\web_driver\\')
from web_driver import *
import time
import requests
from lxml import html

start = time.time()
url = pandas.read_csv('C:\\fake_news\\doc\\note.csv',engine = 'python',sep = ';')
url['text'] = ''
k = 1
note = []
k = 0
for u in url['lien']:
    text_gaia = ''
    page = requests.get(u)
    page = html.fromstring(page.text)
    element = page.xpath('//p//text()')
    for e in element:
        text_gaia += e + ' '
    url['text'].iloc[k] = text_gaia
    k += 1
    print((k,len(url)))
url.to_csv('C:/fake_news/doc/note_text.csv',sep = ';',index = False)