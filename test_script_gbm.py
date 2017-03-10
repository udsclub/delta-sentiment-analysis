import string

import pandas as pd
from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.preprocessing import FunctionTransformer

df = pd.read_table('reviews.csv', header=0, error_bad_lines=False, delimiter='|')
print('review count: {}'.format(len(df)))
# remove duplicates
df.text = df.text.apply(lambda s: BeautifulSoup(s, 'lxml').text)
df.drop_duplicates(subset=['text'], inplace=True)
print('review count, no duplicates: {}'.format(len(df)))

# create train and test sets
X = df['text']
y = df['label']


# preprocessing
def tokenize_text(review):
    word_list = word_tokenize(review)
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list

def find_length(text):
    return len(text)

def transform_sentim(text):
    tdf = pd.DataFrame()
    tdf['text'] = text
    tdf['length'] = tdf.text.apply(lambda s: find_length(s))
    return csr_matrix(tdf[tdf.columns[1:]].values)


extraction_list = []

# 1. custom features
extraction_list.append(['custom_features',
                             FunctionTransformer(func=transform_sentim,
                                                 validate=False,
                                                 accept_sparse=True
                                                )
                            ])
# 2. simple bag-of-words (tf-idf)
stop_words_set = set(stopwords.words('english'))


extraction_list.append(['tfidf',
                             TfidfVectorizer(tokenizer=tokenize_text, ngram_range=(1,2), max_features=20000, stop_words=stop_words_set)
                            ])



# load pre-trained vectorizer
vectorizer = joblib.load('delta_new_vectorizer_2.pkl')
x_transform = vectorizer.transform(X).toarray()

# load pre-trained classifier
gbm = joblib.load('delta_new_model_2.pkl')
y_pred = gbm.predict(x_transform, num_iteration=gbm.best_iteration)

# print scores
print('The rmse of prediction is:', mean_squared_error(y, y_pred) ** 0.5)
print('The accuracy of prediction on test set is:', accuracy_score(y, y_pred))