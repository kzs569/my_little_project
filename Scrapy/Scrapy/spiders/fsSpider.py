import datetime
import traceback
import re
import json
import scrapy
from scrapy import Selector
from scrapy.http import Request
from Scrapy.items import fsSpiderItem
from fake_useragent import UserAgent
import requests

headers = {'User-Agent': UserAgent().random}


class fsSpider(scrapy.Spider):

    def __init__(self, **kwargs):
        self.update_settings == []
        pass

    name = 'fsSpider'

    allow_domains = ['sina.com.cn']

    start_url_list = ['http://feed.mix.sina.com.cn/api/roll/get?pageid=186&lid=1746&num=10&page=1',
                      'http://feed.mix.sina.com.cn/api/roll/get?pageid=186&lid=1746&num=10&page=2',
                      'http://feed.mix.sina.com.cn/api/roll/get?pageid=186&lid=1746&num=10&page=3',
                      'http://feed.mix.sina.com.cn/api/roll/get?pageid=186&lid=1746&num=10&page=4',
                      'http://feed.mix.sina.com.cn/api/roll/get?pageid=186&lid=1746&num=10&page=5'
                      ]

    def start_requests(self):
        for url in self.start_url_list:
            yield Request(url, self.parse)

    def parse(self, response):

        itemlist = []
        try:
            res = json.loads(response.body.decode('utf-8'))
            boxlist = res['result']['data']
            for box in boxlist:
                item = fsSpiderItem()
                title = box['title']
                href = box['url']
                item['title'] = title
                item['href'] = href

                self.log(message='Get title:' + title)
                self.log(message='Get href:' + href)

                print(title, href)
                itemlist.append(item)
        except Exception as error:
            self.log(message='MainPage error:' + traceback.format_exc())

        for item in itemlist:
            yield Request(url=item['href'], meta={'item1': item}, callback=self.parse_sub)

    def parse_sub(self, response):
        item = response.meta['item1']
        r = requests.get(response.url, headers=headers)
        r.encoding = 'utf-8'
        text = r.text.replace(u'&nbsp', u' ')
        try:
            data_source = Selector(text=text).xpath('//div[@class="date-source"]')
            time = data_source.xpath("span[@class='date']/text()")[0].extract()

            line = time.replace('日', '').replace(u'年', '-').replace(u'月', '-')
            standardtime = datetime.datetime.strptime(line.strip(), "%Y-%m-%d %H:%M")
            standardline = standardtime.strftime("%Y-%m-%d %H:%M:%S")
            try:
                source = data_source.xpath("a/text()")[0].extract()
            except IndexError as error:
                source = data_source.xpath("span[@class='source ent-source']/text()")[0].extract()

            contents = Selector(text=text).xpath('//div[@class="article"]/p/text()').extract()
            contents = [cont.strip() for cont in contents]
            contents = '\n'.join(contents)

            print(standardline, source,contents[:50])

            self.log(message="Get time:" + standardline)
            self.log(message="Get source:" + source)
            self.log(message="Get contents:" + contents[:50])

            item['source'] = source
            item['time'] = standardline
            item['contents'] = contents
            yield item
        except Exception as e:
            self.log('SubPage error:')
            self.log(message=traceback.format_exc())


