from preproc import cleaning
import pickle
import pandas as pd

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def tfidfsearch(query):
    tfidfmx = load_obj('tfidf')
    clq = cleaning(query)
    tfidfdataframe = tfidfmx[clq]
    tfidfdataframe['sum'] = tfidfdataframe.select_dtypes(float).sum(1)
    df = tfidfdataframe.sort_values(by=['sum'])
    print