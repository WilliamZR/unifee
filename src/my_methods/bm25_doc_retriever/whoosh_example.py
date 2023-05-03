# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/17 13:52
# Description:

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import os

index_dir = "whoosh_indexdir"
if os.path.exists(index_dir):
    os.system(f"rm -r {index_dir}")
os.mkdir(index_dir)

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
ix = create_in(index_dir, schema)
writer = ix.writer()
writer.add_document(title=u"First document", path=u"/a",
                    content=u"This is the first document we've added!")
writer.add_document(title=u"Second document", path=u"/b",
                    content=u"The second one is even more interesting!")
writer.commit()

with ix.searcher() as searcher:
    query = QueryParser("content", ix.schema).parse("first")
    results = searcher.search(query)
    print(results[0])
