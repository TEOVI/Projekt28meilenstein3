KuaiSAR (v2)
https://zenodo.org/records/8181109

Needed files from the original dataset-directory:
>KuaiSAR_v2
    item_features.csv
    rec_inter.csv

Place these files in the same directory as the python-files that are described in the following.

"dataPreparation_from_item_features.py" extracts the relevant columns from "item_features.csv" and stores them in "item_metadata_w_tags.parquet" to be used by "projekt28-get_samples.py". As this Dataset does not contain text of news-stories BERTopic can not be used to assign each item a topic. The Dataset does provide categories for each item which will be used instead.

"dataPreparation_from_rec_inter.py" extracts the relevant columns from "rec_inter.csv" and stores them in "recommendations.parquet" to be used by "projekt28-get_samples.py". It also only extracts rows (recommendations) which belong to users that have at least a certain number of days with recommendations in the dataset. To change the current minimum number of days required to be eligible edit "dataPreparation_from_rec_inter.py" and change the variable "minimumDaysOnRecord".

"projekt28-aio-KuaiSAR.py" is an All-In-One file to execute the algorithm on the KuaiSAR-(v2)-dataset. At the beginning of the code, a few variables can be changed depending on your needs and means. The folder-structure should be as follows:
>PARENT-FOLDER
    projekt28-aio-KuaiSAR.py
    >data
        item_features.csv
        rec_inter.csv
