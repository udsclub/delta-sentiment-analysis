####MODEL BUILDING PART####

import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib


# data pre-processing
def tokenize_text(text):
    # to lowercase, tokenization
    word_list = re.findall('[A-Za-z]+',text.lower())
    # remowing stopwords
    word_list = [word for word in word_list if word not in stopwords.words('english')]
    # stemming
    word_list = [SnowballStemmer("english").stem(word) for word in word_list]
    return word_list

# feature extraction and vectorization
def build_feature_matrices(X):
    # build a vocabulary that only consider the top 300 features
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, max_features=300)
    # learn vocabulary and return term-document matrix
    X_transform = vectorizer.fit_transform(X).toarray()
    features = vectorizer.get_feature_names()
    save_features(features)
    return X_transform

# saving features
def save_features(features):
    thefile = open('features.txt', 'w')
    for item in features:
        thefile.write("%s\n" % item)
    thefile.close()

# train classifier
def build_clf(X, Y):
    clf = RandomForestClassifier()
    # fit Naive Bayes classifier according to X, y
    clf.fit(X, Y)
    print("Accuracy: ",clf.score(X, Y))
    return clf

# export classifier
def export_trained_clf(clf):
    # compression level for data = 3
    joblib.dump(clf, 'delta_model.pkl', compress = 3)


def main()
    # load and split data
    df = pd.read_table('reviews.csv', header=0, error_bad_lines=False, delimiter='|')
    train, test = train_test_split(df, test_size = 0.3, random_state = 111)
    # vectorize
    X_transform_train = build_feature_matrices(train['text'])
    #build model
    clf = build_clf(X_transform_train, train['label'])
    export_trained_clf(clf)  
    
main()


####MODEL TESTING PART####
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

# data pre-processing
def tokenize_text(text):
    # to lowercase, tokenization
    word_list = re.findall('[A-Za-z]+',text.lower())
    # remowing stopwords
    word_list = [word for word in word_list if word not in stopwords.words('english')]
    # stemming
    word_list = [SnowballStemmer("english").stem(word) for word in word_list]
    return word_list

#vectorization
def build_feature_matrices_test(X):
    # load features
    features_voc = open('features.txt', 'r').read().strip().split('\n')
    # vectorize using loaded features
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, vocabulary = features_voc)
    X_transform = vectorizer.fit_transform(X).toarray()
    return X_transform
    

#predictions
def predict_clf(X, Y):
    clf2 = joblib.load('delta_model.pkl')
    predictions = clf2.predict(X)
    print("Accuracy: ",clf2.score(X, Y))


def main():
    #importing file
    
    ################### тут можно вбырать, какой на каком файле тестить
    ################### если ничего не выбрано - тестить на тренировочном сете в строке 59
    #df = pd.read_table('reviews.csv', header=0, error_bad_lines=False, delimiter='|')
    ###################
    
    #vectorizing data
    X_transform_test = build_feature_matrices_test(test['text'])
    #predictions
    predict_clf(X_transform_test,test['label'])
    
main()
