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
import webapp.my_email as my_email

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

def format_article(raw_article):
  title, link = raw_article[0]
  confidence_score = raw_article[1]
  return {'title': title, 'link': link, 'score': round(confidence_score,2)}





# need to figure out if it's different (maintain state? that would suck)
# send to a slack channel or what?
  # what do I do with the recommendations?
# check if it's already been posted
