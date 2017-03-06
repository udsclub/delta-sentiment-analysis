import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

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
train, test = train_test_split(df, test_size=0.2, random_state=111)
train_text = train['text']
test_text = test['text']

# preprocessing
def tokenize_text(review):
    # tokenize text
    word_list = word_tokenize(review)
    #stemming
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list

stop_words_set = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(tokenizer=tokenize_text, ngram_range=(1,2), max_features=20000, stop_words=stop_words_set)

X_train_features = vectorizer.fit_transform(train_text)
joblib.dump(vectorizer, 'delta_final_vectorizer.pkl', compress = 3)
X_test_features = vectorizer.transform(test_text)

print("Start training ... ")
gbm = lgb.LGBMClassifier(num_leaves=20, learning_rate=0.1, n_estimators = 400)
gbm.fit(X_train_features, train['label'],
        eval_set=[(X_test_features, test['label'])],
        eval_metric='binary_logloss')

print('Start predicting...')
# predict
y_pred_test = gbm.predict(X_test_features, num_iteration=gbm.best_iteration)
y_pred_train = gbm.predict(X_train_features, num_iteration=gbm.best_iteration)
joblib.dump(gbm, 'delta_final_model.pkl', compress = 3)

# eval
print('The rmse of prediction is:', mean_squared_error(test['label'], y_pred_test) ** 0.5)
print('The accuracy of prediction on training set is:', accuracy_score(train['label'], y_pred_train))
print('The accuracy of prediction on test set is:', accuracy_score(test['label'], y_pred_test))

# 0.83
# 0.81
