from joblib import load

text = ["MS Dhoni Is The Fastest Runner Between The Wickets"]

pipeline = load("twitter_sentiment.joblib")

pipeline.predict(text)