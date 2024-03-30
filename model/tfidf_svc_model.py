import numpy as np
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn import linear_model as lm

def getStopwordsList(language):
    stopwords_list = stopwords.words(language)
    stopwords_list.append(" ")
    stopwords_list.append("")
    stopwords_list.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
    stopwords_list.extend([chr(i) for i in range(ord('0'), ord('9') + 1)])
    return stopwords_list

def split_into_words_and_remove_stopwords(text, stopwords):
    # Split text using various separators (space, period, comma)
    words = re.split(r'\s+|[,\.:!?"&]', text)
    # Remove specified stopwords from the list
    words = [word.lower() for word in words if word.lower() not in stopwords]
    return ' '.join(words)

def create_X(data_train, data_test, type='content'):
    X_train = data_train['Review Content']
    X_test = data_test['Review Content']
    if type == 'title':
        X_train = data_train['Review Title']
        X_test = data_test['Review Title']
    elif type == 'both':
        X_train = data_train['Review Content'] + ' ' + data_train['Review Title']
        X_test = data_test['Review Content'] + ' ' + data_test['Review Title']
    return (X_train, X_test)

class TFIDF_SVC_Model:
    def __init__(self, type='svc', kernel='linear'):
        self.model = svm.SVC(kernel=kernel, probability=True)
        if type == 'logistic':
            self.model = lm.LogisticRegression()
            
        self.tfidf = TfidfVectorizer()
            
    def fit(self, X_train, y_train):
        X_train = X_train.apply(lambda x: split_into_words_and_remove_stopwords(x, getStopwordsList('english')))
        remove_list = ['mouse', 'headset', 'anker', 'cable', 'sound', 'quality', 'gaming', 'charging', 'mic', 'wheel', 'scroll', 'razer', 'charge']
        X_train = X_train.apply(lambda x: split_into_words_and_remove_stopwords(x, remove_list))
        X_train = self.tfidf.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        X_test = X_test.apply(lambda x: split_into_words_and_remove_stopwords(x, getStopwordsList('english')))
        remove_list = ['mouse', 'headset', 'anker', 'cable', 'sound', 'quality', 'gaming', 'charging', 'mic', 'wheel', 'scroll', 'razer', 'charge']
        X_test = X_test.apply(lambda x: split_into_words_and_remove_stopwords(x, remove_list))
        X_test = self.tfidf.transform(X_test)
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        X_test = X_test.apply(lambda x: split_into_words_and_remove_stopwords(x, getStopwordsList('english')))
        remove_list = ['mouse', 'headset', 'anker', 'cable', 'sound', 'quality', 'gaming', 'charging', 'mic', 'wheel', 'scroll', 'razer', 'charge']
        X_test = X_test.apply(lambda x: split_into_words_and_remove_stopwords(x, remove_list))
        X_test = self.tfidf.transform(X_test)
        return self.model.predict_proba(X_test)
    
    def predictFromComment(self, comment, type='label'):
        stopwords_list = stopwords.words('english')
        stopwords_list.append(" ")
        stopwords_list.append("")
        stopwords_list.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        stopwords_list.extend([chr(i) for i in range(ord('0'), ord('9') + 1)])
        
        words = re.split(r'\s+|[,\.:!?"&]', comment)
        words = [word.lower() for word in words if word.lower() not in stopwords_list]
        comment = ' '.join(words)
        
        x = np.array([comment])
        x = self.tfidf.transform(x)
        if type == 'proba':
            return self.model.predict_proba(x)[0, 1]
        return 'positive' if self.model.predict_proba(x)[0, 1] >= 0.5 else 'negative'

def objective(trial, data_train, data_test):
    td = trial.suggest_categorical('data-type', ['content', 'title', 'both'])
    tm = trial.suggest_categorical('model-type', ['logistic', 'svc'])
    kr = trial.suggest_categorical('kernel', ['linear', 'poly', 'sigmoid', 'rbf'])
    
    X_train, X_test = create_X(data_train, data_test, td)
    y_train, y_test = data_train['Label'], data_test['Label']
    
    model = Sam_Model(tm, kr)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    scores = f1_score(y_test, y_pred)
    return scores