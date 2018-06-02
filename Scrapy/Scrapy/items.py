# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Field, Item


class CsrcSpiderItem(Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    title = Field()
    full_title = Field()
    href = Field()
    time = Field()
    source = Field()
    content = Field()


class fnSpiderItem(Item):
    title = Field()
    href = Field()
    full_title = Field()
    source = Field()
    time = Field()
    contents = Field()


class hxSpiderItem(Item):
    title = Field()
    time = Field()
    href = Field()
    content = Field()
    source = Field()
