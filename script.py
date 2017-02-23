import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_data():
    df = pd.read_csv('reviews_rt_all.csv', header=0, error_bad_lines=False, delimiter='|')
    return df


# data pre-processing
def tokenize_text(text):
    # delete punctuation symbols
    text_without_punc = ''.join(s for s in text if s not in string.punctuation)
    # tokenize text
    tokenize_text = word_tokenize(text_without_punc)
    stop_list = set(stopwords.words('english'))
    x_without_stop = [word for word in tokenize_text if word.lower() not in stop_list and not word.lower().isdigit()]
    # stemming
    stemmer = SnowballStemmer("english")
    result_text = [stemmer.stem(word) for word in x_without_stop]
    return result_text


# building feature vectors
def build_feature_matrices(df):
    print("Extracting features from the training dataset using a Tfidf vectorizer")
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, max_features=200)
    X = vectorizer.fit_transform(df['text'].values)
    X_array = X.toarray()
    print("n_samples: %d, n_features: %d" % X.shape)
    return X_array


# build classifier
def build_clf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Accuracy: %s' % (score))


def main():
    df = load_data()
    y = df.label.values
    X = build_feature_matrices(df)
    build_clf(X,y)

main()
