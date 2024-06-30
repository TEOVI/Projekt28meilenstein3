# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "insert our paper-name here".
"""

import pandas as pd

df = pd.read_parquet('articles.parquet', engine='pyarrow')

df = df.drop(columns=['title', 'subtitle', 'last_modified_time', 'premium', 'published_time', 'image_ids', 'article_type', 'url', 'ner_clusters', 'entity_groups', 'topics', 'category', 'subcategory', 'total_inviews', 'total_pageviews', 'total_read_time', 'sentiment_score', 'sentiment_label'])

df.rename(columns={"article_id": "item", "body": "text", "category_str": "category"}, inplace = True)

df.to_parquet('item_metadata.parquet', index=False)