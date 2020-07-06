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
note = []
for u in url:
    # text_gaia = ''
    # wds.get_to_page(u)
    # element = wds.find_elements('//p')
    # for e in element:
    #     text_gaia += wds.find_attribute(e,'') + ' '
    # jaime = wds.find_element('//span[@class = "_5n6h _2pih"]')
    # jaime = wds.find_attribute(jaime,'')
    wds.get_to_page('https://www.sharedcount.com/')
    element = wds.find_element('//textarea')
    wds.fill_form(element,u)
    element = wds.find_element('//button[contains(text(),"URL")]')
    wds.click(element)
    start = time.time()
    cont = True
    while time.time() - time.time() < 10 and cont:
        try:
            element = wds.find_element('//td[@class = "tb-data tb-data_face"]')
            cont = False
        except:
            pass
    note.append(wds.find_attribute(element,''))
wds.close_browser()
url = pandas.DataFrame(url)
url.columns = ['lien']
url['note'] = note
url.to_csv('C:/fake_news/doc/note.csv',sep = ';',index = False)