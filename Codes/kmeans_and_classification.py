from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfFileReader
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import numpy as np
import os
import string 
import sys
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

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
        print(root)
        print('\n')
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
    print(len(corpus))
    
    tfidf_matrix,vectorizer = generate_features(corpus)
    num_clusters = 9
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

    #countVectorizer
    X_train, X_test, Y_train, Y_test = train_test_split(corpus, labels, random_state=1, train_size=0.90)
    vect = CountVectorizer(tokenizer = tokenize, stop_words = 'english')
    X_train_dtm = vect.fit_transform(X_train)

    #test
    X_test_dtm = vect.transform(X_test)
    
    #tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_dtm)
    X_test_tfidf = tfidf_transformer.transform(X_test_dtm)
    
    
    
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn import metrics
    
    #MultinomialNB
    mnb = MultinomialNB()
    mnb_t = MultinomialNB()
    mnb.fit(X_train_dtm,Y_train)
    y_pred_class_mnb = mnb.predict(X_test_dtm)
    print(metrics.accuracy_score(Y_test, y_pred_class_mnb) ,"multinomialnb - countvectorizer")
    mnb_t.fit(X_train_tfidf,Y_train)
    y_pred_mnb_t = mnb_t.predict(X_test_tfidf)
    print(metrics.accuracy_score(Y_test,y_pred_mnb_t) ,"multinomialnb - tfidf")
    #grid
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer = tokenize, stop_words = 'english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,cv=5)
    gs_clf.fit(X_train,Y_train)
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    y_nb_grid = gs_clf.predict(X_test)
    print(metrics.accuracy_score(Y_test,y_nb_grid),"grid search- nb")
    '''
    #GaussianNB
    gnb = GaussianNB()
    gnb_t = GaussianNB()
    gnb.fit(X_train_dtm.toarray(),Y_train)
    y_pred_class_gnb = gnb.predict(X_test_dtm.toarray())
    print(metrics.accuracy_score(Y_test, y_pred_class_gnb),"gaussianNB -countvectorizer")
    gnb_t.fit(X_train_tfidf.toarray(),Y_train)
    y_pred_gnb_t = gnb_t.predict(X_test_tfidf.toarray())
    print(metrics.accuracy_score(Y_test,y_pred_gnb_t) ,"gaussianNB - tfidf")
    '''



    #RandomForest
    #grid
    n_estimators = [300,350,250]
    min_samples_split = [2,3,4,5,10]
    random_state =[0,1]
    param_grid = {'n_estimators' : n_estimators ,'min_samples_split' : min_samples_split,'random_state' :random_state}
    random_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    random_grid.fit(X_train_dtm.toarray(), Y_train)
    print(random_grid.best_score_)
    print(random_grid.best_params_)
    y_random_grid = random_grid.predict(X_test_dtm.toarray())
    print(metrics.accuracy_score(Y_test,y_random_grid),"grid search- randomforest")
    #normal
    rdf = RandomForestClassifier(n_estimators=350,random_state=0)
    rdf.fit(X_train_dtm.toarray(), Y_train)
    y_pred_class_rdf = rdf.predict(X_test_dtm.toarray())
    print(metrics.accuracy_score(Y_test, y_pred_class_rdf),"randomForest - Countvectorizer")
    rdf_t = RandomForestClassifier(n_estimators=350,random_state=0)
    rdf_t.fit(X_train_tfidf.toarray(), Y_train)
    y_pred_rdf_t = rdf_t.predict(X_test_tfidf.toarray())
    print(metrics.accuracy_score(Y_test, y_pred_rdf_t),"randomForest - tfidf")
    
    #LogisticRegression
    lr = LogisticRegression(random_state=0)
    lr.fit(X_train_dtm, Y_train)
    y_pred_class_lr =  lr.predict(X_test_dtm)
    print(metrics.accuracy_score(Y_test, y_pred_class_lr),"logisticRegressiion - Countvectorizer")
    lr_t = LogisticRegression(random_state=0)
    lr_t.fit(X_train_tfidf, Y_train)
    y_pred_lr_t = lr_t.predict(X_test_tfidf)
    print(metrics.accuracy_score(Y_test, y_pred_lr_t),"LogisticRegression - tfidf")

    #Linear SVM
    svm_model = LinearSVC()
    svm_model_t = LinearSVC()
    svm_model.fit(X_train_dtm, Y_train)
    y_pred_class_svm_model = svm_model.predict(X_test_dtm)
    print(metrics.accuracy_score(Y_test, y_pred_class_svm_model)," Linear SVM -countvectorizer")
    svm_model_t.fit(X_train_tfidf, Y_train)
    y_pred_svm_model_t = svm_model_t.predict(X_test_tfidf)
    print(metrics.accuracy_score(Y_test, y_pred_svm_model_t),"Linear SVM -tfidf")
    #SVM kernal - rbf
    kernal_svm = SVC()
    kernal_svm_t = SVC()
    kernal_svm.fit(X_train_dtm, Y_train)
    y_pred_kernal_svm = kernal_svm.predict(X_test_dtm)
    print(metrics.accuracy_score(Y_test,y_pred_kernal_svm),"Kernal SVM - Countvectorizer")
    kernal_svm_t.fit(X_train_tfidf,Y_train)
    y_pred_kernal_svm_t = kernal_svm_t.predict(X_test_tfidf)
    print(metrics.accuracy_score(Y_test,y_pred_kernal_svm_t),"Kernal SVM - tfidf")      
    #SVM-SGD
    svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
    svm.fit(X_train_dtm, Y_train)
    y_pred_class_svm = svm.predict(X_test_dtm)
    print(metrics.accuracy_score(Y_test, y_pred_class_svm),"SVM-SGD -countvectorizer")
    svm_t = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
    svm_t.fit(X_train_tfidf, Y_train)
    y_pred_svm_t = svm_t.predict(X_test_tfidf)
    print(metrics.accuracy_score(Y_test, y_pred_svm_t),"SVM-SGD -tfidf")
    #grid
    print("grid")
    from sklearn import svm, grid_search
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel':('poly', 'rbf')}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)
    grid_search.fit(X_train_dtm, Y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    y_grid_search_svm = grid_search.predict(X_test_dtm)
    print(metrics.accuracy_score(Y_test,y_grid_search_svm),"grid search- SVM")
    



    
    '''
    #X_train, X_test, y_train, y_test = train_test_split(corpus, labels, random_state=1,train_size=0.90)
    #X_train_tfidf, vectorizer = generate_features(X_train)
    #X_test_tfidf, vectorizer = generate_features(X_test)

    
    from sklearn.naive_bayes import MultinomialNB,GaussianNB
    from sklearn.metrics import accuracy_score
    clf = MultinomialNB()
    clf.fit(tfidf_matrix,labels)



    
    test_corpus = build_corpus_from_dir('C:/Users/Aksheya/Downloads/Dataset-20180224T085550Z-001/test')
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_test_counts = count_vect.fit_transform(test_corpus)
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
    predicted = clf.predict(X_test_tfidf)
    print(predicted)


    
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(tfidf_matrix,labels)
    predicted = clf.predict(X_test1)
    print(len(predicted))


    
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
    
   
    file_label('.',labels)
    print(len(X_train))
    print(len(X_test))
    print(X_train[0])
    print('\n')
    print(y_train[0])
    print('\n\n\n')
    print(X_train[150])
    print('\n')
    print(y_train[150])
    print('\n\n\n')
    print(X_test[30])
    print('\n')
    print(y_test[30])
    print('\n\n\n')
    print(X_test[70])
    print('\n')
    print(y_test[70])
    print('\n\n\n')

    text_clf_svm = Pipeline([('vect', CountVectorizer()),
    ...                      ('tfidf', TfidfTransformer()),
    ...                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),


    SGDClassifier(loss='hinge', penalty='l2',
    ...                                            alpha=1e-3, n_iter=5, random_state=42)),
    ... ])

	from sklearn.naive_bayes import MultinomialNB,GaussianNB
    from sklearn.metrics import accuracy_score
    clf = MultinomialNB()
    clf.fit(tfidf_matrix,labels)
    

    test_corpus = build_corpus_from_dir('C:/Users/Aksheya/Downloads/Dataset-20180224T085550Z-001/test')
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english', max_features = 1000)
    X_test_tfidf = tfidf_transformer.transform(test_corpus)
    predicted = clf.predict(X_test_tfidf)
    print(predicted)
    '''

