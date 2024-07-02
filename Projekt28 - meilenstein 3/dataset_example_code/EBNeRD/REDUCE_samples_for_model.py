# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "Reproducing the 
Measurement of Filter Bubbles - Machine Learning Internship - Project 28".
"""

import pandas as pd
import numpy as np
np.random.seed(41)

# number of users in the output file
minimumDaysOnRecord = 8
numberOfUsersToKeep = 10000

df = pd.read_parquet('samples_for_model_full.parquet')

df = (df.groupby(['user'])
      .agg({'day_index': lambda x: x.tolist(), 'days_since_signup': lambda x: x.tolist(), 'count': lambda x: x.tolist(), 'variety': lambda x: x.tolist()})
      .reset_index())

df.insert(1, 'days_on_record', df['day_index'].apply(lambda x: len(x)))

reduced_df = df.loc[df['days_on_record'] >= minimumDaysOnRecord]

reduced_df = reduced_df.drop(['days_on_record'], axis=1)

reduced_df = reduced_df.explode(
    ['day_index', 'days_since_signup', 'count', 'variety']).reset_index(drop=True)

reduced_df.to_parquet('samples_for_model_reduced.parquet', index=False)

numUser = reduced_df['user'].nunique()

reduced_df = (reduced_df.groupby(['user'])
              .agg({'day_index': lambda x: x.tolist(), 'days_since_signup': lambda x: x.tolist(), 'count': lambda x: x.tolist(), 'variety': lambda x: x.tolist()})
              .reset_index())

remove_n = int(numUser - numberOfUsersToKeep)
drop_users = np.random.choice(reduced_df.index, remove_n, replace=False)
df_sampled = reduced_df.drop(drop_users)


df_sampled = df_sampled.explode(
    ['day_index', 'days_since_signup', 'count', 'variety']).reset_index(drop=True)

df_sampled.to_parquet('samples_for_model_reduced_sampled.parquet', index=False)
