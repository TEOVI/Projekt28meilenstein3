# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "How Should We Measure Filter
Bubbles? A Regression Model and Evidence for Online News" and then modified by
the authors of the paper "insert our paper-name here" to fit their needs.
"""

# from spacy.lang.en import STOP_WORDS
# from bertopic import BERTopic
# from flair.embeddings import TransformerDocumentEmbeddings
# from hdbscan import HDBSCAN
import pandas as pd
# import numpy as np
# import seaborn as sns
# from sklearn.feature_extraction.text import CountVectorizer
# from umap import UMAP
# from functools import partial
# import json
# import os
# import arviz as az
# import bambi as bmb
import matplotlib.pyplot as plt
# import torch


def main():

    unitVariable = "s"

    """
    The following code works out how many news-stories and under those how many
    different topics are recommended to each user each day (they received at
    least one recommendation).

    INPUT:
        recommendations.parquet
        > file of recommendation-events with unix-timestamp ("timestamp"), user-IDs ("user") 
        and news-story-IDs ("item")
        AND
        item_metadata_w_tags.parquet
        > file of news-stories with news-story-IDs ("item"), text ("text") and 
        topics ("tag")

    OUTPUT:
        samples_for_model.parquet
        > file of entries for each day with user-IDs ("user"), global index
        of the day("day_index"), index of the day since user-sign-up-day 
        ("days_since_signup"), number of reccomendations("count") and number of
        different topics recommended ("variety")
    """

    base_folder = "data"

    # Transform-Recommendation-Log-to-Samples

    recommendations_df = pd.read_parquet(
        f"{base_folder}/recommendations.parquet")
    item_metadata_df = pd.read_parquet(
        f"{base_folder}/item_metadata_w_tags.parquet")

    # Get datetime from epoch
    recommendations_df['datetime'] = pd.to_datetime(
        recommendations_df["timestamp"], unit=unitVariable)

    # Calculate user information
    user_df = pd.to_datetime(recommendations_df.groupby(
        'user').datetime.min().rename('signup_date').dt.date).reset_index()

    user_df.head()

    user_df.to_parquet(
        f'{base_folder}/user_information.parquet', index=True)

    # Add user info to recommendations
    augmented_reco_df = pd.merge(
        recommendations_df, user_df, how="left", on="user", validate="many_to_one")

    # Add date
    augmented_reco_df["date"] = pd.to_datetime(
        augmented_reco_df["datetime"].dt.date)

    augmented_reco_df.head()

    # Assign index to unique days in the dataset
    min_date = augmented_reco_df["date"].min()
    max_date = augmented_reco_df["date"].max()

    min_day = (min_date.isocalendar().week - 1) * \
        7 + min_date.isocalendar().weekday
    min_year = min_date.isocalendar().year
    max_day = (max_date.isocalendar().week - 1) * \
        7 + max_date.isocalendar().weekday
    max_year = max_date.isocalendar().year
    n_days = (max_year - min_year) * 365 + (max_day - min_day) + 1
    min_year, min_day, max_year, max_day, n_days

    day_index_map = {(min_day - 1 + i) % 365 + 1: i for i in range(n_days)}
    # min_day, max_day, day_index_map

    def assign_day_index(x):
        try:
            return day_index_map[x]
        except:
            print('EXCEPT:', x)

    augmented_reco_df["day_index"] = ((augmented_reco_df["date"].dt.isocalendar(
    ).week - 1) * 7 + augmented_reco_df["date"].dt.isocalendar().day).map(assign_day_index)

    # Add days since user signed up
    augmented_reco_df["days_since_signup"] = (
        (augmented_reco_df["date"] - augmented_reco_df["signup_date"]).dt.days).astype(int)

    # Plot how many recommendations were made in each day
    augmented_reco_df["date"].dt.isocalendar().day.hist()
    plt.savefig(f'{base_folder}/recs_each_day.png', dpi='figure')
    plt.clf()

    # How many recommendations made in each day since signup
    augmented_reco_df["days_since_signup"].plot.hist()
    plt.savefig(
        f'{base_folder}/recs_each_day_since_signup.png', dpi='figure')
    plt.clf()

    # Frequency of every day index in the dataset
    augmented_reco_df["day_index"].plot.hist()
    plt.savefig(f'{base_folder}/freq_of_dayIndex.png', dpi='figure')
    plt.clf()

    augmented_reco_df.head()

    # Write intermediate data to folder for reuse
    augmented_reco_df.to_parquet(
        f"{base_folder}/augmented_reco_df.parquet", index=False)

    augmented_reco_df = pd.merge(
        augmented_reco_df, item_metadata_df, on="item")

    augmented_reco_df

    recommendations_grouped_day = augmented_reco_df.groupby(
        ["user", "day_index", "days_since_signup"]).agg({"item": lambda x: len(set(x)), "tag": lambda y: len(set(y))}).reset_index().rename(columns={"item": "count", "tag": "variety"})

    recommendations_grouped_day.to_parquet(
        f"{base_folder}/samples_for_model.parquet",
        index=False)


if __name__ == '__main__':
    main()
