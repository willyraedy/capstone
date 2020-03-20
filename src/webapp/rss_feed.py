import feedparser
from urllib import parse
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from tld import get_tld
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

import os
import sys
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)
import data.tone_analyzer as tone_analyzer
from data.make_dataset import make_media_tag_data

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
  return analyzer.polarity_scores(text)['compound']

def get_tones(text, tone_ids):
  res = tone_analyzer.get(text)
  return [tone_analyzer.extract_score(res, t) for t in tone_ids]

def extract_url_query_param(link):
  raw = parse.parse_qs(parse.urlsplit(link).query)
  if type(raw.get('url')) == list and raw.get('url'):
    return raw.get('url')[0]
  return 'www.google.com'

def strip_html(text_with_tags):
  bs = BeautifulSoup(text_with_tags, features="lxml")
  return bs.text

def fetch_rss_data():
  d = feedparser.parse('https://www.google.com/alerts/feeds/14270752024232241671/6659486510644126588')
  return [
    (strip_html(x['title']), extract_url_query_param(x['link']))
    for x in d['entries']]

def shape_rss_data_for_model(articles):
  df = pd.DataFrame(articles, columns=['title', 'url'])
  df['sentiment_score'] = df.title.map(get_sentiment_score)
  tone_ids = ['analytical', 'tentative']
  tone_scores = df.title.map(lambda t: get_tones(t, tone_ids))
  for idx, tone in enumerate(tone_ids):
    df[tone] = tone_scores.map(lambda x: x[idx])
  file_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'data', 'processed', 'us_media_sources.csv'))
  mc_df = pd.read_csv(file_path)
  df['tld'] = df.url.map(get_tld)
  merged = df.merge(mc_df, how='left', on='tld')
  return merged.fillna(0)

def train_model():
  file_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'data', 'processed', 'prod_training_data.csv'))
  model_data = pd.read_csv(file_path)
  X, y = model_data[[
    'sentiment_score',
    'tag_United States',
    'analytical',
    'tentative',
  ]], model_data.viral
  ros = RandomOverSampler(random_state=42)
  X_fin_ros, y_fin_ros = ros.fit_resample(X, y)
  rf_params = {
    'max_depth': 2,
    'n_estimators': 300,
    'random_state': 42,
    'min_samples_leaf': 17
  }
  rf_final = RandomForestClassifier(**rf_params)
  rf_final.fit(X_fin_ros, y_fin_ros)
  return rf_final

rf = train_model()
articles = fetch_rss_data()
data = shape_rss_data_for_model(articles)
preds = rf.predict_proba(data[[
    'sentiment_score',
    'tag_United States',
    'analytical',
    'tentative',
  ]])

potential = [x for x in zip(articles, preds[:, 1]) if x[1] > 0.5]
in_order = sorted(potential, key=lambda x: -x[1])
top_articles = in_order[:3]
print(top_articles)

# need to figure out if it's different (maintain state? that would suck)
# send to a slack channel or what?
  # what do I do with the recommendations?
# check if it's already been posted
