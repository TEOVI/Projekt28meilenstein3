# -*- coding: utf-8 -*-

"""
This code was written by the authors of the paper "insert our paper-name here".
"""

from datetime import datetime
import pandas as pd

minimumDaysOnRecord = 14

# load rec_inter.csv file into a dataframe
all_recos_df = pd.read_csv("rec_inter.csv")

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
print('Number of users with >=", minimumDaysOnRecord, "days with recommendations on record:',
      chosenUsersCounter)

# create the result-dataframe from the temp. list
chosen_users_df = pd.concat(chosenUsers)

# rename certain columns the fit the recommendations.parquet format
chosen_users_df.rename(
    columns={"user_id": "user", "item_id": "item"}, inplace=True)

# save the relevant columns from the result-dataframe as a csv file
chosen_users_df[["timestamp", "user", "item"]].to_parquet(
    "recommendations.parquet", index=False)
