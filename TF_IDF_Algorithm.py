# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:36:47 2023

@author: user
                                TF-IDF Algorithm
"""
#How to use TF-IDF Algorithm
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
corpus=['The mouse had a tiny little mouse','The cat saw the mouse','The cat catch the mouse','The end of mouse story']
#step1 initialize count vector
cv=CountVectorizer()
#to count the total no.of tf
word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape
#o/p: (4, 11)
#Now next step is to apply IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
#This matrix is in raw matrix,let us convert it in dataframe
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names_out(),columns=["idf_weights"])
#sort ascending
df_idf.sort_values(by=['idf_weights'])
#########################################################
#how to apply tfidf actually to usecase

from sklearn.feature_extraction.text import TfidfVectorizer
corpus=[
        "Thor eatting pizza,loki is eating pizza,Ironman ate pizza already",
        "Apple is announcing new iphone tomorrow",
        "Tesla is announcing new model-3 tomorrow",
        "Google is announcing new pixel-6 tomorrow",
        "Microsoft is announcing new surface tomorrow",
        "Amazon is announcing new eco-dot tomorrow",
        "I am eating biryani and you are eatting graphes"]
#lets create the vectorizer and fit the corpus and transform
#them accordingly
v=TfidfVectorizer()
v.fit