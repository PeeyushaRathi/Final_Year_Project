from PyPDF2 import PdfFileReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import sys
import os

import matplotlib.pyplot as plt 

from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    text = ''
    for i in range(pdf.getNumPages()):
        page = pdf.getPage(i)
        text = text + page.extractText()
    return text

def tokenize(document):
    ps = PorterStemmer()
    '''return words longer than 2 chars and all alpha'''
    tokens = [ps.stem(w) for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens

def build_corpus_from_dir(dir_path):
    corpus = []
    for root, dirs, filenames in os.walk(dir_path,topdown=False):
        for name in filenames:
            if(name.endswith('PDF')):
                #print('\n',name)
                f  = os.path.join(root, name)
                pdf = PdfFileReader(f, "rb")
                document = pdf2text(pdf)
                corpus.append(document)
    return corpus

def file_label(dir_path,labels):
    i=0
    for root, dirs, filenames in os.walk(dir_path, topdown = False):
        for name in filenames:
            if(name.endswith('PDF')):
                print(name, ' = label: ', labels[i])
                i=i+1
                

if __name__ == '__main__':
    corpus = build_corpus_from_dir('.')
    #corpus=["Hi how are you, what you doingg?", "Hey what's up bro? you are cool","Hi what are you up to? Such a cool day"]
    
    vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english', max_features = 250, min_df = 5, max_df = 0.5)
    train_cv = vectorizer.fit_transform(corpus)
    
    a = train_cv.toarray()
    #print('\nThis is A:\n',a)
    b = vectorizer.inverse_transform(a)
    features = vectorizer.get_feature_names()
    #print('\nThis is B:\n',b)
    print('\n\n\n\n\n\n')
    #print(features)
    #print(len(features))
    
    #dist = 1 - cosine_similarity(a)

    num_clusters = 4

    km = KMeans(n_clusters = num_clusters)

    km.fit(a)

    centroids = km.cluster_centers_
    labels = km.labels_

    print("\nTop terms per cluster:\n")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("Cluster %d:" % i),
        cluster_features = []
        for ind in order_centroids[i, :20]:
            #print(' %s' % terms[ind]),
            cluster_features.append(terms[ind]),
        print('\n',cluster_features,'\n')    
        print()

    colors = ["g.","r.","c.","b.","y."]

    for i in range(len(a)):
        #print("\ncoordinate: ", a[i], "      label: ", labels[i])
        plt.plot(a[i])

    plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=150, linewidths = 5, zorder = 10)
    #plt.show()
    
    #print ('\nCentroids:\n',centroids,'\n\n\n')
    #print ('\nLabels:\n',labels,'\n\n\n')

    file_label('.',labels)
