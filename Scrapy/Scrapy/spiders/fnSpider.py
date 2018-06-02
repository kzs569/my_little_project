import datetime
import traceback
import scrapy
from bs4 import BeautifulSoup
from scrapy import Selector
from scrapy.http import Request
from Scrapy.items import fnSpiderItem
from fake_useragent import UserAgent
import requests

headers = {'User-Agent': UserAgent().random}


class fnSpider(scrapy.Spider):

    def __init__(self, **kwargs):
        self.update_settings == []
        pass

    name = 'fnSpider'

    allow_domains = ['financialnews.com.cn']

    start_url_list = ['http://www.financialnews.com.cn/']

    def start_requests(self):
        for url in self.start_url_list:
            yield Request(url, self.parse)

    def parse(self, response):

        itemlist = []
        try:
            infos = Selector(response).xpath('//div[@class="box1"]')
            for info in infos:

                contents2 = info.xpath('//h3/a')
                for cont in contents2:
                    href = cont.xpath('@href')[0].extract()
                    if self.start_url_list[0] not in href:
                        href = self.start_url_list[0] + href[2:]
                    title = cont.xpath('text()')[0].extract()
                    item = fnSpiderItem()
                    item['title'] = title
                    item['href'] = href
                    print(title, href)

                    self.log(message='Get title:' + title)
                    self.log(message='Get href:' + href)

                    itemlist.append(item)
                contents1 = info.xpath('//div[@class="box_list"]/ul/li/a')
                for cont in contents1:
                    href = cont.xpath('@href')[0].extract()
                    if self.start_url_list[0] not in href:
                        href = self.start_url_list[0] + href[2:]
                    title = cont.xpath('text()')[0].extract()

                    self.log(message='Get title:' + title)
                    self.log(message='Get href:' + href)

                    item = fnSpiderItem()
                    item['title'] = title
                    item['href'] = href
                    print(title, href)
                    itemlist.append(item)
        except Exception as e:
            self.log(message='MainPage error:')
            self.log(message=traceback.format_exc())

        print('itemlist length is :', len(itemlist))
        print(itemlist)

        for item in set(itemlist):
            yield Request(url=item['href'], meta={'item1': item}, callback=self.parse_sub)

    def parse_sub(self, response):
        item = response.meta['item1']
        r = requests.get(response.url, headers=headers)
        r.encoding = 'utf-8'
        text = r.text.replace(u'&nbsp', u' ')
        try:
            infos = Selector(text=text).xpath('//div[@class="content"]')
            for info in infos:
                full_title = info.xpath('string(//div[@class="content_title"])')[0].extract()
                source = info.xpath('//div[@class="content_info"]/span/text()')[0].extract()
                time = info.xpath('//div[@class="content_info"]/span/text()')[2].extract()
                standardtime = datetime.datetime.strptime(time[5:].strip(),"%Y-%m-%d %H:%M")
                standardline = standardtime.strftime("%Y-%m-%d %H:%M:%S")
                print(full_title, source, time)
                contents = info.xpath('//div[@class="Custom_UnionStyle"]/p/text()').extract()
                contents = ''.join([cont.strip() for cont in contents])
                print(contents)

                self.log(message='Get full title:' + full_title)
                self.log(message='Get source:' + source)
                self.log(message='Get time:' + standardline)
                self.log(message='Get contents:' + contents)

                item['full_title'] = full_title
                item['source'] = source
                item['time'] = standardline
                item['contents'] = contents
                yield item
        except Exception as e:
            self.log('SubPage error:')
            self.log(message=traceback.format_exc())

