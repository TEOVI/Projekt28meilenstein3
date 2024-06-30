EBNeRD (Ekstra Bladet News Recommendation Dataset)
https://recsys.eb.dk/

Needed files from the original dataset-directory:
>ebnerd_large
    articles.parquet
    >train
        behaviors.parquet

Place these files in the same directory as the python-files that are described in the following.

"dataPreparation_from_articles.py" extracts the relevant columns from "articles.parquet" and stores them in "item_metadata.parquet" to be used by "projekt28-add_topics.py".

"dataPreparation_from_behaviors.py" extracts the relevant columns from "behaviors.parquet" and stores them in "recommendations.parquet" to be used by "projekt28-add_topics.py". It also calculates timestamps from the provided datetimes in the dataset.

"REDUCE_samples_for_model.py" can be used to trim the (from "projekt28-get_samples.py") resulting "samples_for_model.parquet" off users with a low number of days on record. It is also used to reduce the number of users by only keeping a set number of random users out of the (earlier) reduced number of users.

how to user REDUCE_samples_for_model.py:
    1. execute "projekt28-add_topics.py" and "projekt28-get_samples.py"
    2. rename "samples_for_model.parquet" to "samples_for_model_full.parquet" and move it into the same directory as "REDUCE_samples_for_model.py"
    3. edit "REDUCE_samples_for_model.py" variables "numberOfUsersToKeep" and "minimumDaysOnRecord" (or keep the default)
    4. execute "REDUCE_samples_for_model.py"
    5. rename the output file "samples_for_model_reduced_sampled.parquet" to "samples_for_model.parquet", move it into the directory of the original "samples_for_model.parquet" (from step 2.) and continue with "projekt28-fit_model.py"

"projekt28-aio-EBNeRD.py" is an All-In-One file to execute the algorithm on the EBNeRD-dataset. At the beginning of the code, a few variables can be changed depending on your needs and means. The folder-structure should be as follows:
>PARENT-FOLDER
    projekt28-aio-EBNeRD.py
    >data
        articles.parquet
        behaviors.parquet