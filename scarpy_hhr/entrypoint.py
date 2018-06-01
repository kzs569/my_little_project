from scrapy.cmdline import execute

execute(['scrapy', 'crawl', 'csrcSpider'])
#execute(['scrapy', 'crawl', 'fnSpider'])
#execute(['scrapy', 'crawl', 'hxSpider'])

from twisted.internet import reactor
from scrapy.crawler import CrawlerProcess
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scarpy_hhr.spiders import csrcSpider, fnSpider, hxSpider
from scrapy.utils.project import get_project_settings


# def main():
#     configure_logging()
#     settings = get_project_settings()
#
#     process = CrawlerProcess(settings)
#     process.crawl(csrcSpider)
#     process.crawl(fnSpider)
#     process.crawl(hxSpider)
#
#     process.start()
#
#
# if __name__ == '__main__':
#     main()


# from scrapy.cmdline import execute
#
# execute(['scrapy', 'mycrawl'])