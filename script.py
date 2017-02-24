#FINAL SCRIPT TRAIN SET
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize

#loading and splitting data
df = pd.read_table('reviews.csv', sep='|')
train, test = train_test_split(df, test_size = 0.6, random_state = 111)
  
#data pre-processing
def tokenize_text(text):
    #to lowercase, tokenization
    word_list = word_tokenize(text)
    #remowing stopwords
    word_list = [word for word in word_list if word not in stopwords.words('english')]
    # stemming
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list

#feature extraction and vectorization
def build_feature_matrices(df):
    # build a vocabulary that only consider the top 300 features
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, max_features=300)
    # learn vocabulary and return term-document matrix
    X = vectorizer.fit_transform(df['text'].values).toarray()
    features = vectorizer.get_feature_names()
    return X, features

X, features = build_feature_matrices(train)

#write features to the file
thefile = open('features.txt', 'w')
for item in features:
    thefile.write("%s\n" % item)
thefile.close()


# model building





#########################################################################################

#FINAL SCRIPT TEST SET
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

#importing file
df = pd.read_table('reviews.csv', sep='|')

#import feature vector
features_voc = open('features.txt', 'r').read().strip().split('\n')

#tokenize
def tokenize_text(text):
    #to lowercase, tokenization
    word_list = word_tokenize(text)
    #remowing stopwords
    word_list = [word for word in word_list if word not in stopwords.words('english')]
    #stemming
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list

#vectorize
def build_feature_matrices2(df):
    #COuntVecorize using words from loaded vocabulary
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, vocabulary = features_voc)
    X = vectorizer.fit_transform(df['text'].values).toarray()
    return X
#########################################################################################
#IDEAS

import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


def load_data():
    # load dataset into pandas dataframe
    df = pd.read_csv('reviews_rt_all.csv', header=0,
                     error_bad_lines=False, delimiter='|')
    return df


# data pre-processing
def tokenize_text(text):
    # delete punctuation symbols
    punc_symb = string.punctuation
    text_no_punc = ''.join(s for s in text if s not in punc_symb)
    # tokenize text
    token_text = word_tokenize(text_no_punc)
    # delete stop-words
    stop_list = set(stopwords.words('english'))
    clean_text = [w for w in token_text if w.lower() not in stop_list]
    # stemming
    stemmer = SnowballStemmer("english")
    result_text = [stemmer.stem(word) for word in clean_text]
    return result_text


# feature extraction
def build_feature_matrices(df):
    print("Feature extraction using a Tfidf vectorizer")
    # build a vocabulary that only consider the top 200 features
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, max_features=200)
    # learn vocabulary and return term-document matrix
    X = vectorizer.fit_transform(df['text'].values)
    X_array = X.toarray()
    print("n_samples: %d, n_features: %d" % X.shape)
    return X_array


# build classifier
def build_clf(X, y):
    # split dataset into random train and test subsets,
    # the proportion of the dataset to include in the test split - 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = MultinomialNB()
    # fit Naive Bayes classifier according to X, y
    clf.fit(X_train, y_train)
    # return the mean accuracy on the given test data and labels
    score = clf.score(X_test, y_test)
    print('Accuracy: %s' % (score))
    return clf


# dump classifier
def export_trained_clf(clf):
    # compression level for data = 3
    joblib.dump(clf, 'delta_model.pkl', compress=3)


def main():
    df = load_data()
    y = df.label.values
    X = build_feature_matrices(df)
    clf = build_clf(X, y)
    export_trained_clf(clf)

main()
