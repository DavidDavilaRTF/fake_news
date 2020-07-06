import pandas
import numpy
import sys
sys.path.append('C:\\web_driver\\')
from web_driver import *
import time

start = time.time()
url = pandas.read_csv('C:\\fake_news\\doc\\url_gai_final.csv',engine = 'python',sep = ';',header = None)
url = url[0].unique()
k = 1
wds = web_driver_selenium()
wds.create_browser()
for u in url:
    text_gaia = ''
    wds.get_to_page(u)
    element = wds.find_elements('//p')
    for e in element:
        text_gaia += wds.find_attribute(e,'') + ' '
    jaime = wds.find_element('//span[@class = "_5n6h _2pih"]')
    jaime = wds.find_attribute(jaime,'')
wds.close_browser()