import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from joblib import dump


twitter_data = pd.read_csv("/Users/Vitika_Jain/Downloads/DataSets/Twitter_Sentiments/twitter_sentiments.csv")

twitter_data.sample(10)
twitter_data.shape
twitter_data['label'].value_counts()

train, test = train_test_split(twitter_data, test_size=0.2, stratify=twitter_data['label'], random_state=10)

train.shape
test.shape

train['label'].value_counts(normalize=True)
test['label'].value_counts(normalize=True)

tfidf_vectorizer = TfidfVectorizer(lowercase=True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)

tfidf_vectorizer.fit(train['tweet'])

train_idf = tfidf_vectorizer.transform(train['tweet'])
test_idf = tfidf_vectorizer.transform(test['tweet'])

LR_model = LogisticRegression()

LR_model.fit(train_idf, train['label'])

predict_train = LR_model.predict(train_idf)

predict_test = LR_model.predict(test_idf)

f1_score(y_true=train['label'], y_pred=predict_train)
f1_score(y_true=test['label'], y_pred=predict_test)

pipeline = Pipeline(steps=[('tfidf', TfidfVectorizer(lowercase=True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)),
                           ('model', LogisticRegression())])

pipeline.fit(train['tweet'], train['label'])

pipeline.predict(train['tweet'])

text = ["MS Dhoni Is The Fastest Runner Between The Wickets"]

pipeline.predict(text)

dump(pipeline, filename="twitter_sentiment.joblib")
