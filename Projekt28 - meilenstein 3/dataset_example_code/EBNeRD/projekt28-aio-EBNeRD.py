# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "How Should We Measure Filter
Bubbles? A Regression Model and Evidence for Online News" and then modified by
the authors of the paper "insert our paper-name here" to fit their needs.
"""

# from spacy.lang.en import STOP_WORDS
from bertopic import BERTopic
# from flair.embeddings import TransformerDocumentEmbeddings
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np
# import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
# from umap import UMAP
# from functools import partial
# import json
# import os
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
# import torch
import datetime
from datetime import timedelta


def main():

    # random seed used for picking the number of users to keep
    np.random.seed(97)

    # minimum number of days with recommendations for a user to be taken into the model fitting phase
    minimumDaysOnRecord = 8

    # number of users to be taken into the model fitting phase
    numberOfUsersToKeep = 10000

    # unit of timestamps in recommendations.parquet (s/ms)
    unitVariable = "s"

    # name of the folder holding the dataset and all future data
    base_folder = "data"

    starttime = datetime.datetime.now()

    print('script-start - finished', starttime)

    """
    The following code extracts the relevant columns from articles.parquet 
    and creates the item_metadata.parquet-file.
    """

    df = pd.read_parquet(f"{base_folder}/articles.parquet")

    df = df.drop(columns=['title', 'subtitle', 'last_modified_time', 'premium', 'published_time', 'image_ids', 'article_type', 'url', 'ner_clusters',
                 'entity_groups', 'topics', 'category', 'subcategory', 'total_inviews', 'total_pageviews', 'total_read_time', 'sentiment_score', 'sentiment_label'])

    df.rename(columns={"article_id": "item", "body": "text",
              "category_str": "category"}, inplace=True)

    df.to_parquet(f"{base_folder}/item_metadata.parquet", index=False)

    del df

    print('item_metadata.parquet - finished', datetime.datetime.now())

    """
    The following code extracts the relevant columns from behaviors.parquet 
    and creates the recommendations.parquet-file.
    """

    df = pd.read_parquet(f"{base_folder}/behaviors.parquet")

    df = df.drop(columns=['impression_id', 'article_id', 'read_time', 'scroll_percentage', 'device_type', 'article_ids_clicked',
                 'is_sso_user', 'gender', 'postcode', 'age', 'is_subscriber', 'session_id', 'next_read_time', 'next_scroll_percentage'])

    df = df.explode('article_ids_inview')

    df.rename(columns={"impression_time": "timestamp",
              "user_id": "user", "article_ids_inview": "item"}, inplace=True)

    df['timestamp'] = df['timestamp'].apply(lambda x: (
        x - datetime.datetime(1970, 1, 1)) / timedelta(seconds=1))

    df.to_parquet(f"{base_folder}/recommendations.parquet", index=False)

    del df

    print('recommendations.parquet - finished', datetime.datetime.now())

    """
    The following code uses BERTopic to analyse all texts of every news-story
    and work out what topic each news-story is about. Each news-story is
    assigned one topic each.

    INPUT:
        item_metadata.parquet
        > file of news-stories with news-story-IDs ("item") and text ("text")

    OUTPUT:
        item_metadata_w_tags.parquet
        > file of news-stories with news-story-IDs ("item"), text ("text") and topics ("tag")
    """

    az.style.use("arviz-darkgrid")

    # Add-Topics-to-Articles

    base_folder = "data"

    recommended_articles = pd.read_parquet(
        f"{base_folder}/item_metadata.parquet")[['item', 'text']]
    recommended_articles.head()

    # Change to the correct language

    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    topic_model = BERTopic(
        language='multilingual',  # Change this to multilingual when text is non-english
        min_topic_size=10,
        vectorizer_model=CountVectorizer(
            stop_words='english', ngram_range=(1, 2)),
        hdbscan_model=hdbscan_model
    )

    docs = recommended_articles["text"].values
    topics, probs = topic_model.fit_transform(docs)
    recommended_articles["tag"] = topics

    # Store the model so we can use it in the future.
    model_name = "bertopic_base_model"
    topic_model.save(f"{base_folder}/{model_name}")

    recommended_articles[["item", "text", "tag"]].to_parquet(
        f"{base_folder}/item_metadata_w_tags.parquet", index=False)

    print('item_metadata_w_tags.parquet - finished', datetime.datetime.now())

    """
    The following code removes all entries that recommend an article with an assigned topic of "-1"
    """

    df = pd.read_parquet(f"{base_folder}/item_metadata_w_tags.parquet")

    df = df[df.tag != -1]

    df.to_parquet(f"{base_folder}/item_metadata_w_tags.parquet", index=False)

    print('item_metadata_w_tags.parquet (without -1 entries) - finished', datetime.datetime.now())

    article_list = df['item'].to_list()

    df = pd.read_parquet(f"{base_folder}/recommendations.parquet")

    df = df[df['item'].isin(article_list)]

    df.to_parquet(f"{base_folder}/recommendations.parquet", index=False)

    del df

    print('recommendations.parquet (without -1 entries) - finished', datetime.datetime.now())

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

    print('samples_for_model.parquet - finished', datetime.datetime.now())

    """
    The following code keeps only those users that have the before specified 
    minimum number of days of recommendations and among those chooses the 
    specified number of users at random.
    """

    df = pd.read_parquet(f"{base_folder}/samples_for_model.parquet")

    df.to_parquet(f"{base_folder}/samples_for_model_full.parquet", index=False)

    df = (df.groupby(['user'])
          .agg({'day_index': lambda x: x.tolist(), 'days_since_signup': lambda x: x.tolist(), 'count': lambda x: x.tolist(), 'variety': lambda x: x.tolist()})
          .reset_index())

    df.insert(1, 'days_on_record', df['day_index'].apply(lambda x: len(x)))

    df = df.loc[df['days_on_record'] >= minimumDaysOnRecord]

    df = df.drop(['days_on_record'], axis=1)

    df = df.explode(
        ['day_index', 'days_since_signup', 'count', 'variety']).reset_index(drop=True)

    df.to_parquet(
        f"{base_folder}/samples_for_model_reduced.parquet", index=False)

    numUser = df['user'].nunique()

    df = (df.groupby(['user'])
          .agg({'day_index': lambda x: x.tolist(), 'days_since_signup': lambda x: x.tolist(), 'count': lambda x: x.tolist(), 'variety': lambda x: x.tolist()})
          .reset_index())

    remove_n = int(numUser - numberOfUsersToKeep)
    drop_users = np.random.choice(df.index, remove_n, replace=False)
    df = df.drop(drop_users)

    df = df.explode(
        ['day_index', 'days_since_signup', 'count', 'variety']).reset_index(drop=True)

    df.to_parquet(
        f"{base_folder}/samples_for_model.parquet", index=False)

    del df

    print('samples_for_model.parquet (reduced) - finished', datetime.datetime.now())

    """
    The following code defines the model and trains it. The resulting estimation
    for each coefficient is outout at the end.

    INPUT:
        samples_for_model.parquet
        > file of entries for each day with user-IDs ("user"), global index
        of the day("day_index"), index of the day since user-sign-up-day 
        ("days_since_signup"), number of reccomendations("count") and number of
        different topics recommended ("variety")

    OUTPUT:
        exp_coeff_table.parquet
        > file of the exponential of the coefficients "intercept" and 
        "days_since_signup"
        AND
        other_coeff_table.parquet
        > file of the coefficients "Intercept", "days_since_signup", 
        "np.log(count)", "1|day_index_sigma", "1|user_sigma", "variety_alpha"
    """

    # Fit-Model

    df = pd.read_parquet(f"{base_folder}/samples_for_model.parquet")

    # GLMM model configuration
    formula = """
    variety ~ 
    1 
    + (days_since_signup) 
    + (np.log(count)) 
    + (1|user) 
    + (1|day_index)
    """

    # Parameters for the experiment
    draws = 1000
    tune = 3000

    #
    model = bmb.Model(
        formula=formula,
        data=df,
        family="negativebinomial",
        link="log",
        dropna=True,
        auto_scale=True
    )

    trace = model.fit(draws=draws, tune=tune, target_accept=0.9)

    # model.plot_priors()
    # plt.savefig(f'{base_folder}/plot_priors.pdf', dpi=300)

    # az.plot_trace(trace, var_names=["Intercept", "days_since_signup",
    #              "np.log(count)", "1|day_index", "1|day_index_sigma"])
    # plt.savefig(f'{base_folder}/plot_trace.pdf', dpi=300)

    exp_coeff_table = az.summary(
        np.exp(trace.posterior), var_names=[
            "Intercept",
            "days_since_signup",
        ]
    )

    exp_coeff_table

    exp_coeff_table.to_parquet(
        f"{base_folder}/exp_coeff_table.parquet", index=True)

    print('exp_coeff_table.parquet - finished', datetime.datetime.now())

    coeff_table = az.summary(
        trace.posterior, var_names=[
            "Intercept",
            "days_since_signup",
            "np.log(count)",
            "1|day_index_sigma",
            "1|user_sigma",
            "variety_alpha"
        ]
    )

    coeff_table

    coeff_table.to_parquet(f"{base_folder}/other_coeff_table.parquet",
                           index=True)

    print('other_coeff_table.parquet - finished', datetime.datetime.now())

    """
    The following code outputs how long the programm took from start to finish.
    """

    endtime = datetime.datetime.now()
    deltatime = endtime - starttime
    timedf = pd.DataFrame({'starttime': [starttime],
                           'endtime': [endtime],
                           'deltatime': [deltatime]})
    timedf.to_parquet(f"{base_folder}/time_stats.parquet",
                      index=False)


if __name__ == '__main__':
    main()
