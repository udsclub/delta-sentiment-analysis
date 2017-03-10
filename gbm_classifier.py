import numpy as np
import pandas as pd
from nltk import PunktSentenceTokenizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import vaderSentiment
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import FeatureUnion, Pipeline

import lightgbm as lgb
from bs4 import BeautifulSoup

# concatenate dataframes
datasets = ['reviews_clean.csv', 'imdb_small.csv']
df = pd.concat((pd.read_csv(name, engine='c', sep='|',
                 usecols=['label', 'text']) for name in datasets), ignore_index=True)
print('review count: {}'.format(len(df)))

# remove duplicates
df.text = df.text.apply(lambda s: BeautifulSoup(s, 'lxml').text)
df.drop_duplicates(subset=['text'], inplace=True)
print('review count, no duplicates: {}'.format(len(df)))

# create train and test sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)
stop_words_set = set(stopwords.words('english'))

def transform_features(text):
    tdf = pd.DataFrame()
    tdf['text'] = text
    tdf['length'] = tdf.text.apply(lambda s: find_length(s))
    return csr_matrix(tdf[tdf.columns[1:]].values)

# preprocessing
def tokenize_text(review):
    # tokenize text
    word_list = word_tokenize(review)
    #stemming
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list

def find_length(text):
    return len(text)

def sentim_score(text):
    compound_score = 0
    sent_tokenize_list = sent_tokenize(text)
    analyzer = SentimentIntensityAnalyzer()
    for sentence in sent_tokenize_list:
        vs = analyzer.polarity_scores(sentence)
        compound_score += vs.get('compound')
    result = compound_score / len(sent_tokenize_list)
    return result

extraction_list = []

# 1. custom features
extraction_list.append(['custom_features',
                             FunctionTransformer(func=transform_features,
                                                 validate=False,
                                                 accept_sparse=True
                                                )
                            ])
# 2. simple bag-of-words (tf-idf)
extraction_list.append(['tfidf',
                             TfidfVectorizer(tokenizer=tokenize_text, ngram_range=(1,2), max_features=20000, stop_words=stop_words_set)
                            ])

extractor = FeatureUnion(extraction_list)

print("Start training ... ")
gbm = lgb.LGBMClassifier(num_leaves=20, learning_rate=0.1, n_estimators = 400)

X_features = extractor.fit_transform(X_train, y_train)

joblib.dump(extractor, 'delta_new_vectorizer_3.pkl', compress = 3)

print("Start trabsform X_test ... ")
X_test_transform = extractor.transform(X_test)
clf = gbm.fit(X_features, y_train)
print('Start predicting...')
# predict
y_pred_train = clf.predict(X_features)
y_pred_test = clf.predict(X_test_transform)
joblib.dump(clf, 'delta_new_model_3.pkl', compress = 3)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred_test) ** 0.5)
print('The accuracy of prediction on training set is:', accuracy_score(y_train, y_pred_train))
print('The accuracy of prediction on test set is:', accuracy_score(y_test, y_pred_test))
