import string

import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import lightgbm as lgb

# loading data
print("Load data ...")
df = pd.read_table('reviews_clean.csv', header=0, error_bad_lines=False, delimiter='|')
train, test = train_test_split(df, test_size=0.2, random_state=111)
train_text = train['text']
test_text = test['text']


# preprocessing
def tokenize_text(review):
    word_list = word_tokenize(review)
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list

stop_words_set = set(stopwords.words('english') + list(string.punctuation))

vectorizer = TfidfVectorizer(tokenizer=tokenize_text, stop_words=stop_words_set, lowercase=True, max_features=3500)
X_train_features = vectorizer.fit_transform(train_text).toarray()
X_test_features = vectorizer.transform(test_text).toarray()

print("Start training ... ")
gbm = lgb.LGBMClassifier(num_leaves=70, learning_rate=0.1, n_estimators = 300)
gbm.fit(X_train_features, train['label'],
        eval_set=[(X_test_features, test['label'])],
        eval_metric='binary_logloss')

print('Start predicting...')
# predict
y_pred_test = gbm.predict(X_test_features, num_iteration=gbm.best_iteration)
y_pred_train = gbm.predict(X_train_features, num_iteration=gbm.best_iteration)

# eval
print('The rmse of prediction is:', mean_squared_error(test['label'], y_pred_test) ** 0.5)
print('The accuracy of prediction on training set is:', accuracy_score(train['label'], y_pred_train))
print('The accuracy of prediction on test set is:', accuracy_score(test['label'], y_pred_test))

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importances_))

