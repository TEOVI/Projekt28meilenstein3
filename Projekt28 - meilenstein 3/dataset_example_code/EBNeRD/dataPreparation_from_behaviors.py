# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "Reproducing the 
Measurement of Filter Bubbles - Machine Learning Internship - Project 28".
"""

import pandas as pd
import datetime
from datetime import timedelta

df = pd.read_parquet('behaviors.parquet', engine='pyarrow')

df = df.drop(columns=['impression_id', 'article_id', 'read_time', 'scroll_percentage', 'device_type', 'article_ids_clicked', 'is_sso_user', 'gender', 'postcode', 'age', 'is_subscriber', 'session_id', 'next_read_time', 'next_scroll_percentage'])

df = df.explode('article_ids_inview')

df.rename(columns={"impression_time": "timestamp", "user_id": "user", "article_ids_inview": "item"}, inplace = True)

df['timestamp'] = df['timestamp'].apply(lambda x: (x - datetime.datetime(1970, 1, 1)) / timedelta(seconds=1))

df.to_parquet('recommendations.parquet', index=False)
