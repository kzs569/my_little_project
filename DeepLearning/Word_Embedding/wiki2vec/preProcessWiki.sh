#!/bin/bash

#preprocess wiki data
#@kongzishang 2018-04-06

#Extracted corpus from wiki.bz2

#Traditional Chinese to Simplified Chinese
echo "opencc: Traditional Chinese to Simplified Chinese..."
time opencc -i wiki.zh.txt -o wiki.zh.chs.txt -c t2s.json

#Cut words
echo "jieba: Cut words..."
time python -m jieba -d ' ' wiki.zh.chs.txt > wiki.zh.seg.txt

#Change encode
echo "iconv: ascii to utf-8..."
time iconv -c -t UTF-8 < wiki.zh.seg.txt > wiki.zh.seg.utf.txt