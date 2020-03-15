import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import seaborn as sns
from tld import get_tld
from pymongo import MongoClient, ASCENDING
import numpy as np
from data.tone_analyzer import extract_score_from_tones

MONGO_PASSWORD = os.environ['MONGO_USER_PASSWORD']
CROWD_TANGLE = os.environ['CROWD_TANGLE_API_TOKEN']

config = {
  'host': '18.216.52.13:27017',
  'username': 'mongo_user',
  'password': MONGO_PASSWORD,
  'authSource': 'reddit_climate_news'
}

db = MongoClient(**config).reddit_climate_news

def fetch_raw_data(collection, match, fields, drop_cols=['num_comments', 'url']):
    raw = pd.DataFrame.from_records(
        db[collection].find(
            match,
            {'num_comments': 1, 'url': 1, '_id': 1, **fields}
        )
    ).sort_values('num_comments', ascending=False)\
    .drop_duplicates(subset=['url'], keep='first')\
    .drop(columns=drop_cols)

    raw['_id'] = raw._id.map(lambda x: str(x))

    return raw

def make_preceeding_comment_data(collection, preceeding_minutes):
    # add subreddit as a parameter
    comm_bef_df = fetch_raw_data(collection, {}, {'created_utc': 1})

    def get_preceeding_comment_count(timestamp, minutes):
        return db.comments.count_documents({'created_utc': {'$gt': timestamp - 60 * minutes, '$lt': timestamp}})

    comm_bef_df['num_comm_' + str(preceeding_minutes)] = comm_bef_df.created_utc.map(
        lambda t: get_preceeding_comment_count(t, preceeding_minutes))

    return comm_bef_df

def make_preceeding_submission_data(collection, preceeding_minutes):
    sub_bef_df = fetch_raw_data(collection, {}, {'created_utc': 1})

    def get_preceeding_submission_count(timestamp, minutes):
        return db.submissions.count_documents({'created_utc': {'$gt': timestamp - 60 * minutes, '$lt': timestamp}})
    sub_bef_df['num_sub_' + str(preceeding_minutes)] = sub_bef_df.created_utc.map(
        lambda t: get_preceeding_submission_count(t, preceeding_minutes))

    return sub_bef_df

def make_media_tag_data(collection):
    # fetch data
    with open('../data/raw/media_cloud_all_sources.pickle', 'rb') as read_file:
        mc_df = pickle.load(read_file)
    posts_df = fetch_raw_data(
        collection=collection,
        match={},
        fields={},
        drop_cols=['num_comments'])

    # merge data
    posts_df['tld'] = posts_df.url.map(lambda u: get_tld(u, as_object=True).domain)
    mereged_df = posts_df.merge(mc_df, how='inner', on='tld')

    # process
    exploded_df = mereged_df.explode(column='media_source_tags')
    exploded_df['tag'] = exploded_df.media_source_tags.map(lambda t: t['tag'])
    dummied_media = pd.get_dummies(exploded_df[['_id', 'tag']], columns=['tag'])
    media_df = dummied_media.groupby('_id').agg('sum')

    return media_df

def make_coarse_topics_data(collection):
    # fetch data
    topics_raw = fetch_raw_data(
        collection=collection,
        match={'text_razor': {'$exists': True}},
        fields={'text_razor': 1 }
    )

    # remove records where text razor field does not include response
    topics_df = topics_raw[topics_raw.text_razor.map(lambda x: bool(type(x) != str and x.get('response')))]

    topics_df['topics'] = topics_df.text_razor.map(
        lambda x: x.get('response').get('coarseTopics'),
        na_action='ignore')
    topics_df = topics_df.explode('topics')
    topics_df['topic_label'] = topics_df.topics.map(lambda x: x['label'], na_action='ignore')
    topics_df['topic_score'] = topics_df.topics.map(lambda x: x['score'], na_action='ignore')
    topics_df = topics_df.dropna().pivot(index='_id', columns='topic_label', values='topic_score')

    return topics_df

def make_entity_data(collection):
    entities_raw = fetch_raw_data(
        collection=collection,
        match={'text_razor': {'$exists': True}},
        fields={'text_razor': 1}
    )

    # remove records where text razor field does not include response
    entities_df = entities_raw[entities_raw.text_razor.map(lambda x: bool(type(x) != str and x.get('response')))]

    # process
    entities_df['entities'] = entities_df.text_razor.map(
        lambda x: x.get('response').get('entities'),
        na_action='ignore')
    entities_df = entities_df.explode('entities')
    entities_df['entity_label'] = entities_df.entities.map(lambda x: x['entityId'], na_action='ignore')
    entities_df = entities_df.groupby(['_id', 'entity_label']).agg('count').reset_index()
    entities_df = entities_df.dropna().pivot(index='_id', columns='entity_label', values='entities')

    return entities_df

def run(
    collection,
    # subreddit=None,
    viral_threshold=300,
    media_tags=[],
    entities=[],
    topics=[],
    preceeding_activity_minutes=30
):
    # fetch data from db
    aggregate_pipeline = []
    # if subreddit:
    #     aggregate_pipeline.append({'$match': {'subreddit': subreddit}})
    aggregate_pipeline.append({ '$project': {
        'score': 1,
        'num_comments': 1,
        'url': 1,
        'text': '$text_razor.response.cleanedText',
        'created_utc': 1,
        'sentiment_score': 1,
        'tones': '$tone_analyzer.document_tone.tones',
        'fb_interactions': '$crowd_tangle.result.interactions',
        'num_images': { '$cond': { 'if': { '$isArray': "$preview.images" }, 'then': { '$size': "$preview.images" }, 'else': 0} }
    }})

    raw = db[collection].aggregate(aggregate_pipeline)
    raw_df = pd.DataFrame.from_records([{**x, **x.get('fb_interactions', {})} for x in raw])
    raw_df = raw_df.sort_values('num_comments', ascending=False).drop_duplicates(subset=['url'], keep='first')
    print('done with fetching')

    # basic processing
    raw_df['post_date'] = raw_df.created_utc.map(datetime.fromtimestamp)
    raw_df['hour_bucket'] = raw_df.post_date.map(lambda x: x.hour // 4)
    raw_df['day_of_week'] = raw_df.post_date.map(lambda x: x.weekday())
    for t in ['analytical', 'anger', 'confident', 'fear', 'joy', 'sadness', 'tentative']:
        raw_df[t] = raw_df.tones.map(lambda ta: extract_score_from_tones(ta, t))
    raw_df['article_length'] = raw_df.text.map(len, na_action='ignore')
    processed_df = pd.get_dummies(raw_df, columns=['hour_bucket', 'day_of_week'])

    # generate other datasets
    media_df = make_media_tag_data(collection)
    topic_df = make_coarse_topics_data(collection)
    entities_df = make_entity_data(collection)
    num_preceeding_comments_df = make_preceeding_comment_data(collection, preceeding_activity_minutes)
    num_preceeding_submissions_df = make_preceeding_submission_data(collection, preceeding_activity_minutes)
    print('done making other data sets')

    # merge datasets
    processed_df['_id'] = processed_df._id.map(lambda x: str(x))
    processed_df = processed_df.merge(topic_df[topics], how='inner', on='_id')
    processed_df = processed_df.merge(media_df[media_tags], how='inner', on='_id')
    processed_df = processed_df.merge(entities_df[entities], how='inner', on='_id')
    processed_df = processed_df.merge(
        num_preceeding_comments_df[['num_comm_' + str(preceeding_activity_minutes), '_id']],
        how='inner',
        on='_id')
    processed_df = processed_df.merge(
        num_preceeding_submissions_df[['num_sub_' + str(preceeding_activity_minutes), '_id']],
        how='inner',
        on='_id')

    # add additional dependent variables
    processed_df['viral'] = processed_df.num_comments.map(lambda x: 1 if x > viral_threshold else 0)

    return processed_df
