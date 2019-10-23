import pickle
import re
from collections import defaultdict
from string import punctuation

import pandas as pd
import pymorphy2
from nltk.tokenize import word_tokenize


# Сохранение объекта в формате pickle
def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Загрузка объекта в формате pickle
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Очистка текста: Токенизация и Лемматизация
def cleaning(text):
    morph = pymorphy2.MorphAnalyzer()
    text = re.sub(r'[^\w\s]', '', text)
    tokenized = []
    words = word_tokenize(text)
    for word in words:
        p = morph.parse(word)[0]
        tokenized.append(p.normal_form)
    tokenized = [token for token in tokenized if token != " " \
                 and token != "—" \
                 and token != "«" \
                 and token != "»" \
                 and token != ".." \
                 and token.strip() not in punctuation]
    return tokenized


# Открыть табличку CSV, 10000 запросов
def csvopener(file):
    data = pd.read_csv(file)
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('is_duplicate', axis=1)
    data = data.dropna()
    data = data.head(5000)
    return data


# Слияние двух столбцов
def mergetwo(dataf, datas):
    merged = dataf.tolist() + datas.tolist()
    return merged


# Превратить запросы в лист слов
def fromcsvtolist(table):
    dataq1 = table['question1'].apply(cleaning)
    dataq2 = table['question2'].apply(cleaning)
    allqueries = mergetwo(dataq1, dataq2)
    return allqueries


# Финальная функция создания файла с очищенной выдачей
def querycleaner():
    data = csvopener('quora.csv')
    queries = fromcsvtolist(data)
    save_obj(queries, 'query')


# Два словаря: очищенный и неочищенный
def dictimaker():
    data = csvopener('quora.csv')
    cleaned_query = load_obj('query')
    uncleaned_query = mergetwo(data['question1'], data['question2'])
    uncleaned_dictionary = defaultdict(list)
    cleaned_dictionary = defaultdict(list)
    for i in range(len(cleaned_query)):
        uncleaned_dictionary[i].append(uncleaned_query[i])
        cleaned_dictionary[i].append(cleaned_query[i])
    save_obj(uncleaned_dictionary, 'uncldicti')
    save_obj(cleaned_dictionary, 'cldicti')


# Словарь количества вхождений слов в документы
def nqidicti():
    sents = load_obj('query')
    vocab = load_obj('vocab')
    nqidict = defaultdict(int)
    for word in vocab:
        for sent in sents:
            if word in sent:
                nqidict[word] += 1
    save_obj(nqidict, 'nqidict')
