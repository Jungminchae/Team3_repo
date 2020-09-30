import time 
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import quote_plus
from selenium.webdriver.common.keys import Keys
import os

class Insta_img_crawler:
    def __init__(self, tag, driver_path):
        self.baseURL = 'https://www.instagram.com/explore/tags/'
        self.tag = tag 
        self.URL = self.baseURL + self.tag
        self.driver_path = driver_path
        self.driver = Chrome(self.driver_path)
    def URL_open(self):
        # 직접 로그인을 해야 합니다
        self.driver.get(self.URL)
    def page_down(self, page_down_num=1):
        body = self.driver.find_element_by_tag_name('body')
        while page_down_num > 0:
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(1.0)
            page_down_num -=1
    def start_crawling(self):
        if os.path.isdir('./img') != True:
            os.mkdir('./img')
        if os.path.isdir('./img/{}'.format(self.tag)) != True:
            os.mkdir('./img/{}'.format(self.tag))

        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        insta = soup.select('.v1Nh3.kIKUG._bz0w')

        n = 1
        for i in insta:
            time.sleep(0.1)
            # print('https://www.instagram.com'+ i.a['href'])
            imgUrl = i.select_one('.KL4Bh').img['src']
            with urlopen(imgUrl) as f:
                with open('./img/{}/'.format(self.tag) + self.tag + str(n) + '.jpg', 'wb') as h:
                    img = f.read()
                    h.write(img)
            n += 1
            if n % 20 ==0:
                print('{} 이미지 {}장 크롤링 성공'.format(self.tag, n))
                print()

        self.driver.close()