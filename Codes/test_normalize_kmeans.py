from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfFileReader
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import string 
import sys


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


'''generate features of the corpus'''
'''return tfidf_matrix'''
def generate_features(corpus):
    vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english', max_features = 1000, max_df = 0.3, min_df = 7)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix,vectorizer
    
                
if __name__ == '__main__':
    corpus = build_corpus_from_dir('.')
    print(corpus[0])
    '''tfidf_matrix,vectorizer = generate_features(corpus)

    num_clusters = 8
    km = KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    labels = km.labels_

    centroids = km.cluster_centers_
    print("\nTop terms per cluster:\n")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("Cluster %d:" % i),
        cluster_features = []
        for ind in order_centroids[i, :25]:
            #print(' %s' % terms[ind]),
            cluster_features.append(terms[ind]),
        print('\n',cluster_features,'\n')    
        print()


    
    
    joblib.dump(km,  'doc_cluster.pkl')
    km = joblib.load('doc_cluster.pkl')
   
    file_label('.',labels)
    '''
