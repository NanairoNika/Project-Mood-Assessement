import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer


# Сохранение объекта в формате pickle
def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Загрузка объекта в формате pickle
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# из текста TF-IDF
def tf_idf_matrix(texts):
    vectorizer = TfidfVectorizer()
    mx = vectorizer.fit_transform(texts).todense()
    mx = pd.DataFrame(mx, columns=vectorizer.get_feature_names())
    return mx


# Создаём матрицу tf-idf
def making_tfidf():
    sents = load_obj('query')
    strsents = [" ".join(sent) for sent in sents]
    tfidf = tf_idf_matrix(strsents)
    save_obj(tfidf, 'tfidf')


# общий вокабуляр
def making_vocabulary():
    sents = load_obj('query')
    vocab = []
    for sent in sents:
        for word in sent:
            if word not in vocab:
                vocab.append(word)
    vocab = sorted(vocab)
    save_obj(vocab, 'vocab')


# Средняя длина предложений
def avglen(sents):
    counter = 0
    for sent in sents:
        counter += len(sent)
    counter /= len(sents)
    return counter


# Расчёт bm25 для слова
def bm25(sent, word, avgdl, N, nqi, k=2, b=0.75):
    counter = 0
    for el in sent:
        if el == word:
            counter += 1
    TF = counter / len(sent)
    bm = log((N - nqi + 0.5) / nqi + 0.5) * ((TF * (k + 1)) / (TF + k * (1 - b + b * (len(sent) / avgdl))))
    return bm


# Создаём матрицу bm25
def making_bm25():
    sents = load_obj('query')
    vocab = load_obj('vocab')
    nqidict = load_obj('nqidict')
    bm25array = []
    avgdl = avglen(sents)
    numb = len(sents)
    for sent in sents:
        sentarr = []
        for word in vocab:
            nqi = nqidict[word]
            tdbm25 = bm25(sent, word, avgdl, numb, nqi)
            sentarr.append(tdbm25)
        bm25array.append(sentarr)
    bm25array = np.array(bm25array)
    bm25array = pd.DataFrame(bm25array, columns=vocab)
    save_obj(bm25array, 'bm25')


# Нормировка вектора слова
def div_norm(x):
    norm_value = np.sqrt(np.sum(x ** 2))
    if norm_value > 0:
        return x * (1.0 / norm_value)
    else:
        return x


# Получаем вектор предложения
def FT_sentvector(sent, model):
    sent_vector = np.zeros((1, model.vector_size))
    for word in sent:
        if word in model.wv:
            wv = model.wv[word]
            wv = div_norm(wv)
            sent_vector += wv
    sent_vector /= len(sent)
    return sent_vector


# Создаём матрицу FT
def making_FT():
    model_file = 'Fasttextvect/model.model'
    model = KeyedVectors.load(model_file)
    sents = load_obj('query')
    FT_matrix = np.zeros((len(sents), model.vector_size))
    for idx, sent in enumerate(sents):
        sentence_vector = FT_sentvector(sent, model)
        FT_matrix[idx] = sentence_vector
    save_obj(FT_matrix, 'ft')

making_FT()
making_bm25()
making_tfidf()

#def making_ELMO():
