from PyPDF2 import PdfFileReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import sys
import os

from nltk.corpus import stopwords
stop = stopwords.words('english')

def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    text = ''
    for i in range(pdf.getNumPages()):
        page = pdf.getPage(i)
        text = text + page.extractText()
    return text

def stem_tokenize(document):
    '''return stemmed words longer than 2 chars and all alpha'''
    tokens = [stem(w) for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens


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
                f  = os.path.join(root, name)
                pdf = PdfFileReader(f, "rb")
                document = pdf2text(pdf)
                corpus.append(document)
    return corpus

if __name__ == '__main__':
    corpus = build_corpus_from_dir('.')
    #corpus=["Hi how are you, what you doingg?", "Hey what's up bro? you are cool","Hi what are you up to? Such a cool day"]
    vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english', max_features = 20)
    train_cv = vectorizer.fit_transform(corpus)
    a = train_cv.toarray()
    print(a)
    b = vectorizer.inverse_transform(a)
    features = vectorizer.get_feature_names()
    print('\nThis is BBBBBBBBB\n',b)
    print('\n\n\n\n\n\n')
    #print(features)
    #print(len(features))
    #print(corpus.iloc[0])
    #indices = np.argsort(vectorizer.idf_)[::-1]
