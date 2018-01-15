from PyPDF2 import PdfFileReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import sys
import os
from nltk.stem import WordNetLemmatizer
import string 
import matplotlib.pyplot as plt 

from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import preprocessing  # to normalise existing X

def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    text = ''
    for i in range(pdf.getNumPages()):
        page = pdf.getPage(i)
        text = text + page.extractText()
    return text

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemma =[]
    for token in tokens:
        lemma.append(lemmatizer.lemmatize(token))
    return lemma

    
def tokenize(document):
    ps = PorterStemmer()
    '''return words longer than 2 chars and all alpha'''
    tokens = [w.lower() for w in document.split() if len(w) > 2 and w.isalpha() and not w in set(string.punctuation)]
    #print(tokens)
    lemma = lemmatize(tokens)
    return lemma

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
    vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english', max_features = 1000, max_df = 0.3, min_df = 7)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    #print(tfidf_matrix.shape)
    a_array = tfidf_matrix.toarray()
    words = vectorizer.inverse_transform(a_array)
    feature_names = vectorizer.get_feature_names()
    '''
    doc = 0
    feature_index = tfidf_matrix[:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
      print (w, s)
    '''
    dist = cosine_similarity(tfidf_matrix)
    num_clusters = 5
    #print(dist.shape)

    #eigen_values, eigen_vectors = np.linalg.eigh(dist)
    #km = KMeans(n_clusters=3, init='k-means++').fit(eigen_vectors[:,:])

    #print(dist)
    print('\n')
    X_Norm = preprocessing.normalize(tfidf_matrix)
    km = KMeans(n_clusters = num_clusters)
    km.fit(X_Norm)
    centroids = km.cluster_centers_
    labels = km.labels_

    print("\nTop terms per cluster:\n")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("Cluster %d:" % i),
        cluster_features = []
        for ind in order_centroids[i, :30]:
            #print(' %s' % terms[ind]),
            cluster_features.append(terms[ind]),
        print('\n',cluster_features,'\n')    
        print()

    joblib.dump(km,  'doc_cluster.pkl')
    km = joblib.load('doc_cluster.pkl')
    clusters = km.labels_.tolist()
    colors = ["g.","r.","c.","b.","y."]
    '''
    for i in range(len(a_array)):
        plt.plot(a_array[i])

    plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=150, linewidths = 5, zorder = 10)
    plt.show()
    '''
    #print ('\nCentroids:\n',centroids,'\n\n\n')
    #print ('\nLabels:\n',labels,'\n\n\n')
    file_label('.',labels)
