# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "Reproducing the 
Measurement of Filter Bubbles - Machine Learning Internship - Project 28".
"""

import pandas as pd

df = pd.read_csv('item_features.csv')

df = df.drop(columns=['first_level_category_id', 'first_level_category_name', 'second_level_category_id', 'second_level_category_name', 'third_level_category_id', 'third_level_category_name', 'fourth_level_category_id',
             'fourth_level_category_name', 'caption', 'author_id', 'item_type', 'upload_time', 'upload_type', 'second_level_category_name_en', 'third_level_category_name_en', 'fourth_level_category_name_en'])

df.rename(columns={"item_id": "item",
          "first_level_category_name_en": "tag"}, inplace=True)

df['text'] = ''

df.to_parquet('item_metadata_w_tags.parquet', index=False)
