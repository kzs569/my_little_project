# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
from scrapy import Request
import logging
from scrapy import settings
import hashlib
import requests



class ImagescrapyPipeline(object):
    def process_item(self, item, spider):
        return item


class ImageDownloadPipeline(ImagesPipeline):
    loggger = logging.getLogger("ImagePipelineDownloadLogger")

    headers = {
        'accept': 'image/webp,image/*,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.6',
        'connection': 'keep-alive',
        'host': 'i.meizitu.net',
        "referer": "http://m.mzitu.com/137371/41",
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36"
    }

    def handle_redirect(self, file_url):
        response = requests.head(file_url)
        if response.status_code == 302:
            file_url = response.headers["Location"]
        return file_url

    def get_media_requests(self, item, info):
        for image_url in item['image_urls']:
            image_url = self.handle_redirect(image_url)
            yield Request(image_url, headers=self.headers)

    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        if not image_paths:
            raise DropItem("Item contains no images")
        item['image_paths'] = settings.get('IMAGES_STORE') + '\\full\\' + hashlib.sha1(item['href']).hexdigest()
        #self.loggger.log(msg='存储路径' + str(dict(item)))
        return item

    def file_path(self, request, response=None, info=None):
        """
        :param request: 每一个图片下载管道请求
        :param response:
        :param info:
        :param strip :清洗Windows系统的文件夹非法字符，避免无法创建目录
        :return: 每套图的分类目录
        """
        item = request.meta['item']
        folder = item['title']
        folder_strip = folder.strip()
        image_guid = request.url.split('/')[-1]
        filename = u'full/{0}/{1}'.format(folder_strip, image_guid)
        return filename

    # def image_key(self, url):
    #     # year, month = url.split('/')[-3], url.split('/')[-2]
    #     image_guid = hashlib.sha1(url).hexdigest()
    #     # img_path = "%s/%s/%s" % (year, month, self.title)
    #     return '%s.jpg' % (image_guid)

