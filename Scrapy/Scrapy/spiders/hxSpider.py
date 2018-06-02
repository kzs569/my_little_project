import datetime
import traceback
import re
import scrapy
from scrapy import Selector
from scrapy.http import Request
from Scrapy.items import hxSpiderItem
from fake_useragent import UserAgent
import requests

headers = {'User-Agent': UserAgent().random}


class fnSpider(scrapy.Spider):

    def __init__(self, **kwargs):
        self.update_settings == []
        self.pattern = re.compile(
            "(?:(?!0000)[0-9]{4}-(?:(?:0[1-9]|1[0-2])-(?:0[1-9]|1[0-9]|2[0-8])|(?:0[13-9]|1[0-2])-(?:29|30)|(?:0[13578]|1[02])-31)|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)-02-29)\s+([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]")
        pass

    name = 'hxSpider'

    allow_domains = ['hexun.com']

    start_url_list = ['http://insurance.hexun.com/bxhyzx/index.html']

    def start_requests(self):
        for url in self.start_url_list:
            yield Request(url, self.parse, encoding='gb2312')

    def parse(self, response):

        itemlist = []
        try:
            infos = Selector(response).xpath('//div[@class="temp01"]/ul/li')

            for info in infos:
                item = hxSpiderItem()
                href = info.xpath('a/@href')[0].extract()
                title = info.xpath('a/text()')[0].extract()

                self.log(message='Get title:' + title)
                self.log(message='Get href:' + href)

                print(title, href)

                item['href'] = href
                item['title'] = title
                itemlist.append(item)
        except Exception as error:
            self.log(message='MainPage error:')
            self.log(message=traceback.format_exc())

        for item in itemlist:
            yield Request(url=item['href'], meta={'item1': item}, callback=self.parse_sub)

    def parse_sub(self, response):
        item = response.meta['item1']
        r = requests.get(response.url, headers=headers)
        r.encoding = 'gb2312'
        text = r.text.replace(u'&nbsp', u' ')
        try:
            infos = Selector(text=text).xpath('//div[@class="tip fl"]')[0]

            time = infos.xpath('span/text()')[0].extract()
            standardtime = datetime.datetime.strptime(time.strip(), "%Y-%m-%d %H:%M:%S")
            standardline = standardtime.strftime("%Y-%m-%d %H:%M:%S")

            source = infos.xpath('a/text()')[0].extract()
            contents = Selector(text=text).xpath('//div[@class="art_contextBox"]/p/text()').extract()
            contents = ''.join([info.strip() for info in contents])
            self.log(message='Get contents:' + contents)
            print(time, source, contents)
            item['time'] = standardline
            item['source'] = source
            item['content'] = contents
            yield item
        except IndexError as e:
            infos = Selector(text=text).xpath('string(//div[@class="tip fl"])')[0].extract()

            time = re.search(self.pattern, infos).group()
            standardtime = datetime.datetime.strptime(time.strip(), "%Y-%m-%d %H:%M:%S")
            standardline = standardtime.strftime("%Y-%m-%d %H:%M:%S")

            source = re.sub(pattern=self.pattern, repl='', string=infos)
            source = source.replace('\r', '').replace('\n', '').replace('\t', '')

            contents = Selector(text=text).xpath('//div[@class="art_contextBox"]/p/text()').extract()
            contents = ''.join([info.strip() for info in contents])
            self.log(message='Get contents:' + contents)
            print(time, source, contents)
            item['time'] = standardline
            item['source'] = source
            item['content'] = contents
            yield item
        except Exception as error:
            self.log(message='SubPage error:')
            self.log(message=traceback.format_exc())
