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
import numpy as np
# import seaborn as sns
# from sklearn.feature_extraction.text import CountVectorizer
# from umap import UMAP
# from functools import partial
# import json
# import os
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
# import torch
from datetime import datetime


def main():

    minimumDaysOnRecord = 14

    # unit of timestamps in recommendations.parquet (s/ms)
    unitVariable = "ms"

    # name of the folder holding the dataset and all future data
    base_folder = "data"

    starttime = datetime.now()

    print('script-start - finished', starttime)

    """
    The following code extracts the relevant columns from item_features.csv 
    and creates the item_metadata.parquet-file.
    """

    df = pd.read_csv(f"{base_folder}/item_features.csv")

    df = df.drop(columns=['first_level_category_id', 'first_level_category_name', 'second_level_category_id', 'second_level_category_name', 'third_level_category_id', 'third_level_category_name', 'fourth_level_category_id',
                 'fourth_level_category_name', 'caption', 'author_id', 'item_type', 'upload_time', 'upload_type', 'second_level_category_name_en', 'third_level_category_name_en', 'fourth_level_category_name_en'])

    df.rename(columns={"item_id": "item",
              "first_level_category_name_en": "tag"}, inplace=True)

    df['text'] = ''

    df.to_parquet(f"{base_folder}/item_metadata_w_tags.parquet", index=False)

    print('item_metadata_w_tags.parquet - finished', datetime.now())

    """
    The following code extracts the relevant columns from rec_inter.csv 
    and creates the recommendations.parquet-file.
    """

    # load rec_inter.csv file into a dataframe
    all_recos_df = pd.read_csv(f"{base_folder}/rec_inter.csv")

    all_recos_df = all_recos_df

    # define a function to convert a datetime into only a date

    def datetime_to_date(dt):
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        return dt.date()

    # execute said function on the coloum "time" to turn every entry into a date
    all_recos_df['time'] = all_recos_df['time'].apply(datetime_to_date)

    # group all recommendations by the user that received them
    all_recos_grouped_by_user_and_time_df = all_recos_df.groupby("user_id")

    # create temporary list to store chosen users
    chosenUsers = []

    # create temporary variable to count the number of chosen users
    chosenUsersCounter = 0

    # go through every user-group created before
    for name, group in all_recos_grouped_by_user_and_time_df:

        # group the recommendations for this user by date
        current_user = group.groupby("time")

        # check if the user has >= minimumDaysOnRecord days with recommendations on record
        if current_user.ngroups >= minimumDaysOnRecord:

            # save this users recommendations in the temp. list
            chosenUsers.append(group)

            # increment the temp. counter
            chosenUsersCounter += 1

    # output a short info-message
    print('Number of users with >=', minimumDaysOnRecord, 'days with recommendations on record:',
          chosenUsersCounter)

    # create the result-dataframe from the temp. list
    chosen_users_df = pd.concat(chosenUsers)

    # rename certain columns the fit the recommendations.parquet format
    chosen_users_df.rename(
        columns={"user_id": "user", "item_id": "item"}, inplace=True)

    # save the relevant columns from the result-dataframe as a csv file
    chosen_users_df[["timestamp", "user", "item"]].to_parquet(
        f"{base_folder}/recommendations.parquet", index=False)

    print('recommendations.parquet - finished', datetime.now())

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

    print('samples_for_model.parquet - finished', datetime.now())

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
        ],round_to=6
    )

    exp_coeff_table

    exp_coeff_table.to_parquet(
        f"{base_folder}/exp_coeff_table.parquet", index=True)

    print('exp_coeff_table.parquet - finished', datetime.now())

    coeff_table = az.summary(
        trace.posterior, var_names=[
            "Intercept",
            "days_since_signup",
            "np.log(count)",
            "1|day_index_sigma",
            "1|user_sigma",
            "variety_alpha"
        ],round_to=6
    )

    coeff_table

    coeff_table.to_parquet(f"{base_folder}/other_coeff_table.parquet",
                           index=True)

    print('other_coeff_table.parquet - finished', datetime.now())

    """
    The following code outputs how long the programm took from start to finish.
    """

    endtime = datetime.now()
    deltatime = endtime - starttime
    timedf = pd.DataFrame({'starttime': [starttime],
                           'endtime': [endtime],
                           'deltatime': [deltatime]})
    timedf.to_parquet(f"{base_folder}/time_stats.parquet",
                      index=False)


if __name__ == '__main__':
    main()
