# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "How Should We Measure Filter
Bubbles? A Regression Model and Evidence for Online News" and then modified by
the authors of the paper "insert our paper-name here" to fit their needs.
"""

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

az.style.use("arviz-darkgrid")


def main():

    base_folder = "data"

    recommended_articles = pd.read_csv(
        f"{base_folder}/item_metadata.csv")[['item', 'text']]
    recommended_articles.head()

    # Change to the correct language
    # from spacy.lang.en import STOP_WORDS

    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    topic_model = BERTopic(
        language='english',  # Change this to multilingual when text is non-english
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

    # remove all entries with a -1 tag
    recommended_articles = recommended_articles[recommended_articles.tag != -1]

    recommended_articles[["item", "text", "tag"]].to_csv(
        f"{base_folder}/item_metadata_w_tags.csv", index=False)

    recommendations_df = pd.read_csv(f"{base_folder}/recommendations.csv")
    item_metadata_df = pd.read_csv(f"{base_folder}/item_metadata_w_tags.csv")

    # Get datetime from epoch
    recommendations_df['datetime'] = pd.to_datetime(
        recommendations_df["timestamp"], unit="s")

    # Calculate user information
    user_df = pd.to_datetime(recommendations_df.groupby(
        'user').datetime.min().rename('signup_date').dt.date).reset_index()

    user_df.head()

    user_df.to_csv(f'{base_folder}/user_information.csv',
                   index=True, header=True)

    # Add user info to recommendations
    augmented_reco_df = pd.merge(
        recommendations_df, user_df, how="left", on="user", validate="many_to_one")

    # Add date
    augmented_reco_df["date"] = pd.to_datetime(
        augmented_reco_df["datetime"].dt.date)

    augmented_reco_df.head()

    # Change signup_date and date to first day of that week, so that we can easily calculate the weeks_since_signup
    augmented_reco_df['date'] = augmented_reco_df['date'] - \
        augmented_reco_df['date'].dt.weekday * np.timedelta64(1, 'D')
    augmented_reco_df['signup_date'] = augmented_reco_df['signup_date'] - \
        augmented_reco_df['signup_date'].dt.weekday * np.timedelta64(1, 'D')

    augmented_reco_df.head()

    # Assign index to unique weeks in the dataset

    min_date = augmented_reco_df["date"].min()
    max_date = augmented_reco_df["date"].max()

    min_week = min_date.isocalendar().week
    min_year = min_date.isocalendar().year
    max_week = max_date.isocalendar().week
    max_year = max_date.isocalendar().year
    n_weeks = (max_year - min_year) * 52 + (max_week - min_week) + 1
    min_year, min_week, max_year, max_week, n_weeks

    week_index_map = {(min_week - 1 + i) % 52 + 1: i for i in range(n_weeks)}
    # min_week, max_week, week_index_map

    def assign_week_index(x):
        try:
            return week_index_map[x]
        except:
            print(x)

    augmented_reco_df["week_index"] = augmented_reco_df["date"].dt.isocalendar(
    ).week.map(assign_week_index)

    # Add weeks since user signed up
    augmented_reco_df["weeks_since_signup"] = (
        (augmented_reco_df["date"] - augmented_reco_df["signup_date"]).dt.days / 7).astype(int)

    # Plot how many recommendations were made in each week
    augmented_reco_df["date"].dt.isocalendar().week.hist()

    # How many recommendations made in each week since signup
    augmented_reco_df["weeks_since_signup"].plot.hist()

    # Frequency of every week index in the dataset
    augmented_reco_df["week_index"].plot.hist()

    augmented_reco_df.head()

    # Write intermediate data to folder for reuse
    augmented_reco_df.to_csv(
        f"{base_folder}/augmented_reco_df.csv", index=False)

    augmented_reco_df = pd.merge(
        augmented_reco_df, item_metadata_df, on="item")

    augmented_reco_df

    recommendations_grouped_week = augmented_reco_df.groupby(
        ["user", "week_index", "weeks_since_signup"]).agg({"item": lambda x: len(set(x)), "tag": lambda y: len(set(y))}).reset_index().rename(columns={"item": "count", "tag": "variety"})

    recommendations_grouped_week.to_csv(
        f"{base_folder}/samples_for_model.csv",
        index=False,
        header=True,
    )

    df = pd.read_csv(f"{base_folder}/samples_for_model.csv")

    # GLMM model configuration
    formula = """
    variety ~ 
    1 
    + (weeks_since_signup) 
    + (np.log(count)) 
    + (1|user) 
    + (1|week_index)
    """

    # Parameters for the experiment
    draws = 1000
    tune = 3000

    #
    model = bmb.Model(
        formula,
        df,
        family="negativebinomial",
        link="log",
        dropna=True,
        auto_scale=True
    )

    trace = model.fit(draws=draws, tune=tune, target_accept=0.9)

    model.plot_priors()
    plt.savefig(f'{base_folder}/plot_priors.pdf', dpi=300)

    az.plot_trace(trace, var_names=["Intercept", "weeks_since_signup",
                  "np.log(count)", "1|week_index", "1|week_index_sigma"])
    plt.savefig(f'{base_folder}/plot_trace.pdf', dpi=300)

    exp_coeff_table = az.summary(
        np.exp(trace.posterior), var_names=[
            "Intercept",
            "weeks_since_signup",
        ]
    )

    exp_coeff_table

    exp_coeff_table.to_csv(
        f"{base_folder}/exp_coeff_table.csv", header=True, index=True)

    coeff_table = az.summary(trace.posterior, var_names=[
        "Intercept",
        "weeks_since_signup",
        "np.log(count)",
        "1|week_index_sigma",
        "1|user_sigma",
        "variety_alpha"
    ])

    coeff_table

    coeff_table.to_csv(
        f"{base_folder}/other_coeff_table.csv", header=True, index=False)


if __name__ == '__main__':
    main()
