#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robin Kong (kzs569@gmail.com)
# Copyrigh 2018

from __future__ import print_function

import logging
import os.path
import six
import sys
from optparse import OptionParser

from gensim.corpora import WikiCorpus


def main():
    #返回path最后的文件名。如何path以／或\结尾，那么就会返回空值。
    #本质是os.path.split(path)的第二个元素
    program = os.path.basename(sys.argv[0])

    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python3 preProcessWiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if six.PY3:
            output.write(b' '.join(text).decode('utf-8') + '\n')
        else:
            output.write(space.join(text) + "\n")
        i = i + 1
        if i % 10000 == 0:
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="ifilename",
                      help="input path to the corpus")
    parser.add_option("-o", "--output", dest="ofilename",
                      help="output path to the processed corpus")
    (options,args) = parser.parse_args(sys.argv)
    main()
