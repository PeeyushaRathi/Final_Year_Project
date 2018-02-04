from PyPDF2 import PdfFileReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import sys
import os
import string 
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import preprocessing  # to normalise existing X
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster,dendrogram,linkage
from sklearn import manifold, datasets


'''iterate over pages and extract text'''
'''return extracted text'''
def pdf2text(pdf):
    text = ''
    for i in range(pdf.getNumPages()):
        page = pdf.getPage(i)
        text = text + page.extractText()
    return text


'''lemmatize tokens'''
'''return lemma'''
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemma =[]
    for token in tokens:
        lemma.append(lemmatizer.lemmatize(token))
    return lemma


'''tokenize document'''
'''return lemmatized tokens'''
def tokenize(document):
    '''return words longer than 2 chars and all alpha'''
    tokens = [w.lower() for w in document.split() if len(w) > 2 and w.isalpha() and not w in set(string.punctuation)]
    lemma = lemmatize(tokens)
    return lemma


'''create a corpus of text documents'''
'''return a 1D array of text corpus'''
def build_corpus_from_dir(dir_path):
    corpus = []
    for root, dirs, filenames in os.walk(dir_path,topdown=False):
        for name in filenames:
            if(name.endswith('PDF')):
                f  = os.path.join(root, name)
                pdf = PdfFileReader(f, "rb")
                document = pdf2text(pdf)
                corpus.append(document)
    return corpus


'''print file and its label'''
def file_label(dir_path,labels):
    i=0
    for root, dirs, filenames in os.walk(dir_path, topdown = False):
        for name in filenames:
            if(name.endswith('PDF')):
                print(name, ' = label: ', labels[i])
                i=i+1

                

if __name__ == '__main__':
    corpus = build_corpus_from_dir('.')
    vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english', max_features = 1000, max_df = 0.3, min_df = 7)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    #print(tfidf_matrix.shape)
    a_array = tfidf_matrix.toarray()
    words = vectorizer.inverse_transform(a_array)
    feature_names = vectorizer.get_feature_names()

    dist = cosine_similarity(tfidf_matrix)

    linkage_matrix = linkage(dist, 'ward') #define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, leaf_rotation = 90.);

    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    
    plt.show()

    max_d = 14
    clusters = fcluster(linkage_matrix, max_d, criterion = 'distance')
    print(clusters)
    
    



