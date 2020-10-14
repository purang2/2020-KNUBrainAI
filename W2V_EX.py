# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:11:51 2020

@author: puran
@title: 유튜브 BERT강의 중 Word2Vec 실습. 
@link: https://www.youtube.com/watch?v=qlxrXX5uBoU
"""
from gensim.models.word2vec import Word2Vec
import gensim

path ='train_corpus.txt'
sentences = gensim.models.word2vec.Text8Corpus(path)

model = Word2Vec(sentences, min_count=5, size=100, window=5)
model.save('w2v_model')

saved_model=Word2Vec.load('w2v_model')

word_vector = saved_model['강아지']

saved_model.similarity('강아지','멍멍이')

saved_model.similar_by_word('강아지')