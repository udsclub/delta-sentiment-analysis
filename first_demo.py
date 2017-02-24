import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

# loading and splitting data
df = pd.read_table('reviews_rt_all.csv', header=0, error_bad_lines=False, delimiter='|')
train, test = train_test_split(df, test_size=0.6, random_state=111)


# data pre-processing
def tokenize_text(text):
    # to lowercase, tokenization
    word_list = word_tokenize(text)
    # remowing stopwords
    word_list = [word for word in word_list if word not in stopwords.words('english')]
    # stemming
    stemmer = SnowballStemmer("english")
    word_list = [stemmer.stem(word) for word in word_list]
    return word_list


# feature extraction and vectorization
def build_feature_matrices(X_train):
    # build a vocabulary that only consider the top 300 features
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, max_features=300)
    # learn vocabulary and return term-document matrix
    X_train_transform = vectorizer.fit_transform(X_train).toarray()
    features = vectorizer.get_feature_names()
    save_features(features)
    return X_train_transform


# train classifier
def build_clf(X_train, y_train):
    clf = RandomForestClassifier()
    # fit Naive Bayes classifier according to X, y
    clf.fit(X_train, y_train)
    return clf


# dump classifier
def export_trained_clf(clf):
    # compression level for data = 3
    joblib.dump(clf, 'delta_model.pkl', compress=3)


def save_features(features):
    thefile = open('features.txt', 'w')
    for item in features:
        thefile.write("%s\n" % item)
    thefile.close()


X_train_transform = build_feature_matrices(train['text'])
clf = build_clf(X_train_transform, train['label'])
export_trained_clf(clf)