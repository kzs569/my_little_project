import datetime
import traceback

import scrapy
from scrapy import Selector
from scrapy.http import Request
from scarpy_hhr.items import CsrcSpiderItem
from fake_useragent import UserAgent
import requests
import logging as log

headers = {'User-Agent': UserAgent().random}


class csrcSpider(scrapy.Spider):

    def __init__(self, **kwargs):
        self.update_settings == []
        pass

    name = 'csrcSpider'

    allow_domains = ['csrc.gov.cn']

    start_url_list = ['http://www.csrc.gov.cn/pub/newsite/zjhxwfb/xwdd']

    def start_requests(self):
        for url in self.start_url_list:
            yield Request(url, self.parse)

    def parse(self, response):

        itemlist = []
        try:
            infos = Selector(response).xpath('//ul[@id="myul"]/li')
            for info in infos:
                item = CsrcSpiderItem()
                href = info.xpath('a/@href')[0].extract()
                title = info.xpath('a/text()')[0].extract()
                # time = info.xpath('span/text()')[0].extract()
                href = self.start_url_list[0] + href[1:]

                self.log(message='Get title:' + title)
                # self.log(message='Get time:' + time)
                self.log(message='Get href:' + href)

                item['title'] = title
                item['href'] = href
                # item['time'] = time

                itemlist.append(item)
        except Exception as error:
            self.log(message='MainPage error:')
            self.log(message=traceback.format_exc())

        for item in itemlist:
            yield Request(url=item['href'], meta={'item1': item}, callback=self.parse_sub)

    def parse_sub(self, response):
        item = response.meta['item1']
        r = requests.get(response.url, headers=headers)
        r.encoding = 'utf-8'
        text = r.text.replace(u'&nbsp', u' ')
        try:
            infos = Selector(text=text).xpath('//div[@class="content"]')
            for info in infos:
                full_title = info.xpath('//div[@class="title"]/text()')[0].extract()
                source = info.xpath('//div[@class="time"]/span/text()')[0].extract()
                time = info.xpath('//div[@class="time"]/span/text()')[1].extract()
                standardtime = datetime.datetime.strptime(time[3:].strip(), "%Y-%m-%d")
                standardline = standardtime.strftime("%Y-%m-%d %H:%M:%S")
                print(full_title, source, time)
                contents = info.xpath('//div[@class="Custom_UnionStyle"]/p/span/text()').extract()
                contents = ''.join([cont.strip() for cont in contents])
                print(contents)
                self.log(message='Get href:' + contents)
                item['full_title'] = full_title
                item['source'] = source
                item['time'] = standardline
                item['content'] = contents
                yield item
        except Exception as error:
            self.log(message='SubPage error:')
            self.log(message=traceback.format_exc())
