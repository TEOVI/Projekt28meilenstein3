# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "How Should We Measure Filter
Bubbles? A Regression Model and Evidence for Online News" and then modified by
the authors of the paper "Reproducing the Measurement of Filter Bubbles - Machine 
Learning Internship - Project 28" to fit their needs.
"""

# from spacy.lang.en import STOP_WORDS
from bertopic import BERTopic
# from flair.embeddings import TransformerDocumentEmbeddings
from hdbscan import HDBSCAN
import pandas as pd
# import numpy as np
# import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
# from umap import UMAP
# from functools import partial
# import json
# import os
import arviz as az
# import bambi as bmb
# import matplotlib.pyplot as plt
# import torch


def main():
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

    base_folder = "data"

    az.style.use("arviz-darkgrid")

    # Add-Topics-to-Articles

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
            stop_words='multilingual', ngram_range=(1, 2)),
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


if __name__ == '__main__':
    main()
