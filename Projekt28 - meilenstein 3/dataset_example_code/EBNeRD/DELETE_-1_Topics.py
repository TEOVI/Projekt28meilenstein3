# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "insert our paper-name here".
"""

import pandas as pd

df = pd.read_parquet('item_metadata_w_tags.parquet')

df = df[df.tag != -1]

df.to_parquet('item_metadata_w_tags_without_-1.parquet', index=False)


article_list = df['item'].to_list()

df_reco = pd.read_parquet('recommendations.parquet')

df_reco = df_reco[df_reco['item'].isin(article_list)]

df_reco.to_parquet('recommendations_without_-1.parquet', index=False)
