from dotenv import load_dotenv
load_dotenv()
import os
import requests
import datetime
import time
from pymongo import MongoClient
import numpy as np

MONGO_PASSWORD = os.environ['MONGO_USER_PASSWORD']

config = {
  'host': '3.20.206.120:27017',
  'username': 'mongo_user',
  'password': MONGO_PASSWORD,
  'authSource': 'reddit_climate_news'
}

db = MongoClient(**config).reddit_climate_news

def get_all(subreddit, start_utc, end_utc):
  assert start_utc > end_utc, 'Scrapes backwards so start_utc must be larger than end_utc'

  curr_utc = start_utc
  retries = 0
  total_scraped = 0
  while curr_utc > end_utc:
    try:
      # get submissions from same range as comments
      res_sub = requests.get(f'https://api.pushshift.io/reddit/search/submission/?sort=desc&subreddit={subreddit}&limit=500&before={curr_utc}')
      submissions = res_sub.json()['data']
      next_utc = submissions[-1]['created_utc']

      sub_to_insert = [x for x in submissions if int(x['created_utc']) > end_utc]
      if sub_to_insert:
        total_scraped += len(sub_to_insert)
        print(total_scraped, ' - ', next_utc)
        db.submissions.insert_many(sub_to_insert)

      curr_utc = next_utc
      time.sleep(0.3)
    except Exception as e:
      print(e)
      retries += 1
      if retries > 3:
        raise e
