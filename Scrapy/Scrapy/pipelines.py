# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


import traceback

import pymysql
from scrapy.conf import settings
import os
import datetime
import redis
import logging
from contextlib import contextmanager

from scrapy import signals
from scrapy.exporters import JsonItemExporter
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
from sqlalchemy.orm import sessionmaker
from Scrapy.items import CsrcSpiderItem, fnSpiderItem, hxSpiderItem, fsSpiderItem

HOST = settings.get("MYSQL_HOST")
PORT = settings.get("MYSQL_PORT")
DB = settings.get("MYSQL_DBNAME")
USER = settings.get("MYSQL_USER")
PASSWD = settings.get("MYSQL_PASSWD")


class ScrapyPipeline(object):
    logger = logging.getLogger("PipelingLog")
    logger.setLevel(level="DEBUG")

    def __init__(self):
        self.connect = pymysql.connect(host=HOST,
                                       db=DB,
                                       user=USER,
                                       # passwd=PASSWD,
                                       charset='utf8',
                                       use_unicode=True)
        self.cursor = self.connect.cursor()

    def process_item(self, item, spider):

        if item.__class__ == CsrcSpiderItem:
            try:
                self.cursor.execute("""CREATE TABLE IF NOT EXISTS `csrc` (
                                      `title` varchar(45) DEFAULT NULL,
                                      `href` varchar(200) NOT NULL,
                                      `time` DATETIME DEFAULT NULL,
                                      `source` varchar(45) DEFAULT NULL,
                                      `content` TEXT(65535) DEFAULT NULL,
                                      PRIMARY KEY (`href`)
                                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8  COLLATE=utf8_general_ci;""")
                self.cursor.execute("""select * from csrc where href = %s;""", item['href'])
                ret = self.cursor.fetchone()
                if ret:
                    self.cursor.execute(
                        """update csrc set title = %s,href = %s,time = %s,
                            source = %s,content = %s
                            where href = %s;""",
                        (item['title'],
                         item['href'],
                         item['time'],
                         item['source'],
                         item['content'],
                         item['href']))
                else:
                    self.cursor.execute(
                        """insert into csrc(title,href,time,source,content)
                          value (%s,%s,%s,%s,%s);""",
                        (item['title'],
                         item['href'],
                         item['time'],
                         item['source'],
                         item['content']))
                self.connect.commit()
                print('QAQ ----> csrc 正在写入数据')
                self.logger.log(msg="csrc写入数据" + str(dict(item)), level=10)
            except Exception as e:
                self.logger.log(msg="csrc sql error" + traceback.format_exc(), level=40)
        elif item.__class__ == fnSpiderItem:
            try:
                self.cursor.execute("""CREATE TABLE if not exists fn (
                                      `title` VARCHAR(45) NULL,
                                      `href` VARCHAR(200) NOT NULL,
                                      `full_title` VARCHAR(45) NULL,
                                      `source` VARCHAR(45) NULL,
                                      `time` DATETIME NULL,
                                      `contents` TEXT(65535) NULL,
                                      PRIMARY KEY (`href`))
                                      ENGINE=InnoDB DEFAULT CHARSET=utf8  COLLATE=utf8_general_ci;
                                    """)
                self.cursor.execute("""select * from fn where href = %s""", item['href'])
                ret = self.cursor.fetchone()
                if ret:
                    self.cursor.execute(
                        """update fn set title = %s,href = %s,time = %s,
                            source = %s,contents = %s,full_title = %s
                            where href = %s""",
                        (item['title'],
                         item['href'],
                         item['time'],
                         item['source'],
                         item['contents'],
                         item['full_title'],
                         item['href']))
                else:
                    self.cursor.execute(
                        """insert into fn(title,href,full_title,source,time,contents)
                          value (%s,%s,%s,%s,%s,%s);""",
                        (item['title'],
                         item['href'],
                         item['full_title'],
                         item['source'],
                         item['time'],
                         item['contents']))
                self.connect.commit()
                print('QAQ ----> fn 正在写入数据')
                self.logger.log(msg="fn 写入数据" + str(dict(item)), level=10)
            except Exception as e:
                self.logger.log(msg="fn sql error" + traceback.format_exc(), level=40)
        elif item.__class__ == hxSpiderItem:
            try:
                self.cursor.execute("""CREATE TABLE if not exists hx (
                                      `title` VARCHAR(45) NULL,
                                      `href` VARCHAR(200) NOT NULL,
                                      `time` DATETIME NULL,
                                      `source` VARCHAR(45) NULL,
                                      `content` TEXT(65535) NULL,
                                      PRIMARY KEY (`href`))
                                      ENGINE=InnoDB DEFAULT CHARSET=utf8  COLLATE=utf8_general_ci;
                                    """)
                self.cursor.execute("""select * from hx where href = %s""", item['href'])
                ret = self.cursor.fetchone()
                if ret:
                    self.cursor.execute(
                        """update hx set title = %s,href = %s,time = %s,
                            content = %s,source = %s
                            where href = %s""",
                        (item['title'],
                         item['href'],
                         item['time'],
                         item['content'],
                         item['source'],
                         item['href']))
                else:
                    self.cursor.execute(
                        """insert into hx(title,href,time,source,content)
                          value (%s,%s,%s,%s,%s)""",
                        (item['title'],
                         item['href'],
                         item['time'],
                         item['source'],
                         item['content']))
                self.connect.commit()
                print('QAQ ----> hx 正在写入数据')
                self.logger.log(msg="hx 写入数据" + str(dict(item)), level=10)
            except Exception as e:
                self.logger.log(msg="hx sql error" + traceback.format_exc(), level=40)
        elif item.__class__ == fsSpiderItem:
            try:
                self.cursor.execute("""CREATE TABLE if not exists fs (
                                      `title` TEXT(10000) NULL,
                                      `href` VARCHAR(200) NOT NULL,
                                      `time` DATETIME NULL,
                                      `source` VARCHAR(45) NULL,
                                      `content` TEXT(65535) NULL,
                                      PRIMARY KEY (`href`))
                                      ENGINE=InnoDB DEFAULT CHARSET=utf8  COLLATE=utf8_general_ci;
                                    """)
                self.cursor.execute("""select * from fs where href = %s""", item['href'])
                ret = self.cursor.fetchone()
                if ret:
                    self.cursor.execute(
                        """update fs set title = %s,href = %s,time = %s,
                            content = %s,source = %s
                            where href = %s""",
                        (item['title'],
                         item['href'],
                         item['time'],
                         item['contents'],
                         item['source'],
                         item['href']))
                else:
                    self.cursor.execute(
                        """insert into fs(title,href,time,source,content)
                          value (%s,%s,%s,%s,%s)""",
                        (item['title'],
                         item['href'],
                         item['time'],
                         item['source'],
                         item['contents']))
                self.connect.commit()
                print('QAQ ----> fs 正在写入数据')
                self.logger.log(msg="fs 正在写入数据" + str(dict(item)), level=10)
            except Exception as e:
                self.logger.log(msg="fs sql error" + traceback.format_exc(), level=40)
        return item
