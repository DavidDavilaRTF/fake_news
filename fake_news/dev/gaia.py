import pandas
import numpy
import sys
sys.path.append('C:\\web_driver\\')
from web_driver import *

wds = web_driver_selenium()
wds.create_browser()
wds.get_to_page('https://lumieresurgaia.com/sitemap.xml')
element = wds.find_elements('//a[contains(@href,"gaia")]')
url = []
for e in element:
    url.append(wds.find_attribute(e,'href'))
url_gaia = []
for u in url:
    wds.get_to_page(u)
    element = wds.find_elements('//a[contains(@href,"gaia")]')
    for e in element:
        url_gaia.append(wds.find_attribute(e,'href'))
wds.close_browser()
url_gaia = pandas.DataFrame(url_gaia)
url_gaia.columns = ['FkN']
url_gaia.to_csv('C:\\fake_news\\doc\\url_gai.csv',sep = ';',index = False)