from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import os
import pandas as pd

def train_model():
  file_path = os.path.abspath(
    os.path.join(__file__, '..', '..', '..', 'data', 'processed', 'prod_training_data.csv'))
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
