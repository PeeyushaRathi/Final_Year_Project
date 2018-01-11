import nltk
import string
import os
import PyPDF2 
import textract
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#path = '\PIYUSHA\Desktop\FYP\Dataset\Rape\Rape-Bombay\POCSO'

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    print("here")
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
      
for subdir, dirs, files in os.walk(".", topdown=False):
    for file in files:
        if(file != 'test_tfidf.py'):
            file_path = subdir + os.path.sep + file
            pdfFileObj = open(file_path,'rb')
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            num_pages = pdfReader.numPages
            count = 0
            text = ""
            while count < num_pages:
                pageObj = pdfReader.getPage(count)
                count +=1
                text += pageObj.extractText()
            if text != "":
               text = text
            else:
               text = textract.process(fileurl, method='tesseract', language='eng')

            tokens = word_tokenize(text)
            punctuations = ['(', ')', ',' , '.', '!', '#', '&', '*', '?', '/','-','_',';']
            stop_words = stopwords.words('english')
            keywords = [word for word in tokens if not word in stop_words and not word in punctuations]
            token_dict[file] = keywords
        
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict)    

print (tfs)
print ('\n')
print ('\n')
print ('\n')

