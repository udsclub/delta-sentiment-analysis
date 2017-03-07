import os
import re
import string

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib


def is_english(s):
    words = s.split()
    non_english = 0
    for w in words:
        try:
            w.encode('ascii')
        except UnicodeEncodeError:
            non_english += 1
    return True if non_english*1.0/len(words) <= 0.05 else False


def tokenize_text(review):
    word_list = word_tokenize(review)
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list

df1 = pd.read_table('reviews_rt_all.csv', header=0, error_bad_lines=False, delimiter='|')
df1.drop_duplicates(subset=['text'], inplace=True)

print(len(df1.index))
df1 = df1[df1['text'].apply(is_english)]
print(len(df1.index))

train1, test1 = train_test_split(df1, test_size=0.2, random_state=111)

train_text1 = train1['text']
test_text1 = test1['text']

stop_words_set = set(stopwords.words('english') + list(string.punctuation))
# stop_words_set = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(tokenizer=tokenize_text,
                             stop_words=stop_words_set,
                             lowercase=True,
                             max_features=20000)
X_train_features1 = vectorizer.fit_transform(train_text1)

print('trained vectorizer in memory')

print(vectorizer.get_feature_names())

# joblib.dump(vectorizer, 'models/tfidf/2000.pkl')

X_test_features1 = vectorizer.transform(test_text1)
# X_test_features2 = vectorizer.transform(test_text2).toarray()

print('test set vectorized')

filepath = 'models/random-forest/'
directory = os.path.dirname(filepath)
if not os.path.exists(directory):
    os.makedirs(directory)

# clf = ExtraTreesClassifier(n_estimators=100,
#                            # min_samples_leaf=10,
#                            # max_features=200,
#                            random_state=49, verbose=2)
clf = AdaBoostClassifier(n_estimators=2000,
                         # learning_rate=0.2,
                           random_state=49)
clf.fit(X_train_features1, train1['label'])

print('random forest trained')

print("Accuracy train: ",clf.score(X_train_features1, train1['label']))
print("Accuracy test: ",clf.score(X_test_features1, test1['label']))

joblib.dump(clf, filepath + '/adaboost-2000-tfidf-20000.pkl', compress=3)

# joblib.dump(clf, filepath + '/randforest-extra-tree-100-min-samples-leaf-20-tfidf-1000.pkl', compress=3)
