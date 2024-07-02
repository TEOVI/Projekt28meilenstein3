# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "How Should We Measure Filter
Bubbles? A Regression Model and Evidence for Online News" and then modified by
the authors of the paper "Reproducing the Measurement of Filter Bubbles - Machine 
Learning Internship - Project 28" to fit their needs.
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
# import matplotlib.pyplot as plt
# import torch
import datetime


def main():
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

    base_folder = "data"

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

    print('exp_coeff_table.parquet - finished', datetime.datetime.now())

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


if __name__ == '__main__':
    main()
