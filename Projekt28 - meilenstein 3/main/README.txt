First you need to provide the needed data. Refer to the README files in the dataset_example_code-directories to know how to do that on the dataset EBNeRD and KuaiSAR.
After you provided "recommendations.parquet" and "item_metadata.parquet" you may use "projekt28-aio.py". We would recommend to take each step at a time to be able to react and adjust to problems as they occur. To do that, execute the following files in that order:
    1. "projekt28-add_topics.py"
    2. "projekt28-get_samples.py"
    3. "projekt28-fit_model.py"
Using the EBNeRD Dataset - after Step 2. - you can/should use "REDUCE_samples_for_model.py". Refer to the corresponding README for detailed information.

Be aware that you may need to change the unit used to get the datetime-format from the epoch-timestamp-format ("s" (seconds) or "ms" (milliseconds)) depending on the timestamp format present in "recommendations.parquet". To do this, simply edit "projekt28-get_samples.py" (or "projekt28-aio.py") and change the variable "unitVariable".