import os
import sys
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)
from models.train_model import train_model
import email_tool.rss_feed as rss_feed
import email_tool.my_email as my_email

rf = train_model()
articles = rss_feed.fetch_rss_data()
data = rss_feed.shape_rss_data_for_model(articles)
preds = rf.predict_proba(data[[
    'sentiment_score',
    'tag_United States',
    'analytical',
    'tentative',
  ]])

potential = [x for x in zip(articles, preds[:, 1]) if x[1] > 0.5]
in_order = sorted(potential, key=lambda x: -x[1])
top_articles = in_order[:3]
formatted_articles = [rss_feed.format_article(x) for x in top_articles]

my_email.send_climate_articles('wraedy@gmail.com', formatted_articles)
