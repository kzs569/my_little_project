import datetime
import traceback
import re
import scrapy
from scrapy import Selector
from scrapy.http import Request
from ImageScrapy.items import ImagescrapyItem
from fake_useragent import UserAgent
import requests

headers = {'User-Agent': UserAgent().random}


class imageSpider(scrapy.Spider):

    def __init__(self, **kwargs):
        self.update_settings == []
        self.pattern = re.compile(u"[0-9]/([0-9]*?)页")
        pass

    name = 'imageSpider'

    allow_domains = ['mzitu.com']

    start_url_list = ['http://www.mzitu.com/']

    def start_requests(self):
        for url in self.start_url_list:
            yield Request(url, self.parse)

    def parse(self, response):

        itemlist = []
        try:
            infos = Selector(response).xpath('//div[@class="archive-brick"]')
            for info in infos:
                item = ImagescrapyItem()
                href = info.xpath("a[@class='clear']/@href")[0].extract()
                title = info.xpath("string(a)")[0].extract()
                self.log(message='Get title:' + title)
                self.log(message='Get href:' + href)

                item['title'] = title
                item['href'] = href

                itemlist.append(item)
        except Exception as error:
            self.log(message='MainPage error:')
            self.log(message=traceback.format_exc())

        item = ImagescrapyItem()
        item['href'] = "http://m.mzitu.com/137371"
        itemlist.append(item)

        for item in itemlist:
            yield Request(url=item['href'], meta={'item1': item}, callback=self.parse_sub)

    def parse_sub(self, response):


        item = response.meta['item1']
        r = requests.get(response.url, headers=headers)
        r.encoding = 'utf-8'
        text = r.text.replace(u'&nbsp', u' ')
        try:
            maxpage = Selector(text=text).xpath('//div[@class="prev-next"]/span/text()')[0].extract()
            print("maxpage:" + maxpage)
            maxpage = re.search(self.pattern, maxpage).group(1)
            maxpage = int(maxpage)
            self.log(message='Get maxpage:' + str(maxpage))
            for i in range(1, maxpage):
                item['imagepage'] = item['href'] + '/' + str(i)
                yield Request(url=item['imagepage'], meta={'item2': item}, callback=self.parse_image)
        except Exception as error:
            self.log(message='SubPage error:')
            self.log(message=traceback.format_exc())

    def parse_image(self, response):
        item = response.meta['item2']
        r = requests.get(response.url, headers=headers)
        r.encoding = 'utf-8'
        text = r.text.replace(u'&nbsp', u' ')
        try:
            image_urls = Selector(text=text).xpath('//div[@class="place-padding"]/figure/p/a/img/@src')[0].extract()

            item['image_urls'] = [image_urls]
            print(image_urls)
            self.log(message="Get url:" + image_urls)
            yield item
        except Exception as error:
            self.log(message='ImagePage error:')
            self.log(message=traceback.format_exc())

