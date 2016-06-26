# -*- coding: utf-8 -*-
import graphlab, jieba
import re, chardet, codecs, sys

def test_data_process():
    sentences = []
    fp = codecs.open("./data/news_tensite_xml.dat", encoding="GB18030")
    for line in fp.readlines():
        m = re.match(r"<content>(.*?)</content>", line.strip())
        if m:
            sentence = str(m.group(1).encode("utf8"))
            if sentence.strip() != "":
                seg_list = jieba.cut(sentence, cut_all=False)
                sentences.append(" ".join(seg_list))
    fp.close()
    s = graphlab.SArray(sentences)
    features = graphlab.text_analytics.count_words(s)
    test_data = graphlab.SFrame()
    test_data['feature'] = features
    test_data['X2'] = sentences
    return test_data

if __name__ == "__main__":
    test_data = test_data_process()
    for sentence in test_data['X2']:
        print sentence
