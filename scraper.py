import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from multiprocessing import Process, Queue, Pool
import threading
import sys
from time import sleep


def get_data(i,url,class_attr,folder):    
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64;     x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate",     "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    r = requests.get(url, headers=headers)
    content = r.content
    soup = BeautifulSoup(content)
    image_divs = soup.findAll('div', attrs={'class':class_attr})

    for image in image_divs:
        i+=1
        try:
            img_tag = image.find('img')
            img_url = img_tag['src']
        except:
            continue
        save_image(img_url, folder + str(i) + ".jpg")

    sleep(5)
    return i


def save_image(image_url, filename):
    response = requests.get(image_url, stream=True)
    if not response.ok:
        print(response)
        return

    img_data = response.content
    with open(filename, 'wb') as handler:
        handler.write(img_data)

    print(filename)



i=310
data=[

["https://www.amazon.in/s?k=formal+shirt+diverse&page=1",'a-section aok-relative s-image-tall-aspect',"formal_shirt/",],
["https://www.amazon.in/s?k=formal+shirt+diverse&page=2",'a-section aok-relative s-image-tall-aspect',"formal_shirt/",],
["https://www.amazon.in/s?k=formal+shirt+diverse&page=3",'a-section aok-relative s-image-tall-aspect',"formal_shirt/",],
["https://www.amazon.in/s?k=formal+shirt+diverse&page=4",'a-section aok-relative s-image-tall-aspect',"formal_shirt/",],
["https://www.amazon.in/s?k=formal+shirt+diverse&page=5",'a-section aok-relative s-image-tall-aspect',"formal_shirt/",],
["https://www.amazon.in/s?k=formal+shirt+diverse&page=6",'a-section aok-relative s-image-tall-aspect',"formal_shirt/",],


]

for url in data:
    print(url[0])
    i = get_data(i, *url)
