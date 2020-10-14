# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:38:25 2020

@author: puran
@title: WordEmbedding FastText유튜브실습

"""
from gensim.models.word2vec import FastText
import gensim

path = 'train_corpus.txt'
sentences = gensim.models.word2vec.Text8Corpus(path)

model = FastText(sentences,min_count=5, size=100, window=5)
model.save('ft_model')

saved_model = FastText.load('ft_model')

word_vector = saved_model['강아지']

saved_model.similarity('강아지','멍멍이')

saved_model.similar_by_word('강아지')
