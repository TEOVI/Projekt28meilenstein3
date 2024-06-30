First you need to provide the needed data. Refer to the README files in the dataset_example_code-directories to know how to do that on the dataset EBNeRD and KuaiSAR.
After you provided "recommendations.parquet" and "item_metadata.parquet" you may use "projekt28-aio.py". We would recommend to take each step at a time to be able to react and adjust to problems as they occur. To do that, execute the following files in that order:
    1. "projekt28-add_topics.py"
    2. "projekt28-get_samples.py"
    3. "projekt28-fit_model.py"
Using the EBNeRD Dataset - after Step 2. - you can/should use "REDUCE_samples_for_model.py". Refer to the corresponding README for detailed information.

Be aware that you may need to change the unit used to get the datetime-format from the epoch-timestamp-format ("s" (seconds) or "ms" (milliseconds)) depending on the timestamp format present in "recommendations.parquet". To do this, simply edit "projekt28-get_samples.py" (or "projekt28-aio.py") and change the variable "unitVariable".


To create a conda-environment containing the needed packages to run these files, follow these steps:
1.  download an install a conda distribution (e.g. miniconda)

execute the following commands:
2.  conda create -n projekt28-env python=3.12.4

3.  conda activate projekt28-env

4.  conda install m2w64-toolchain

5.  conda install -c conda-forge hdbscan

edit and use the following command to move to the working-directory
6.  cd C:\...\Projekt28 - meilenstein 3\main

7.  pip install -r requirements.txt

now you can execute the python files in the current directory - for example the aio-file
    python projekt28-aio.py
