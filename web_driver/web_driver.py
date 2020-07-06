from selenium import webdriver
import time
import pandas
from selenium.webdriver.common.keys import Keys

class web_driver_selenium:
    def __init__(self):
        self.driver = None

    def create_browser(self):
        opts = webdriver.ChromeOptions()
        opts.add_argument('--start-maximized')
        opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246")
        self.driver = webdriver.Chrome(executable_path = 'C:\\web_driver\\chromedriver.exe',options=opts)

    def get_to_page(self,url):
        self.driver.get(url)

    def close_browser(self):
        self.driver.quit()

    def close_currently_page(self):
        self.driver.close()

    def find_element(self,xpath):
        return self.driver.find_element_by_xpath(xpath=xpath)
    
    def find_attribute(self,element,attr):
        if attr != '':
            return element.get_attribute(attr)
        else:
            return element.text

    def find_elements(self,xpath):
        return self.driver.find_elements_by_xpath(xpath=xpath)

    def click(self,element):
        element.click()
    
    def fill_form(self,element,message):
        element.send_keys(message)

    def get_current_url(self):
        return self.driver.current_url
    
    def screenshot(self,filename_path):
        self.driver.save_screenshot(filename_path)
