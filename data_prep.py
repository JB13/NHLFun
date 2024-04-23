import pandas as pd
import numpy as np

import torch


"""
Index(['Unnamed: 0', 'Game_Id', 'Date', 'Period', 'Event', 'Description',
       'Time_Elapsed', 'Seconds_Elapsed', 'Strength', 'Ev_Zone', 'Type',
       'Ev_Team', 'Home_Zone', 'Away_Team', 'Home_Team', 'p1_name', 'p1_ID',
       'p2_name', 'p2_ID', 'p3_name', 'p3_ID', 'awayPlayer1', 'awayPlayer1_id',
       'awayPlayer2', 'awayPlayer2_id', 'awayPlayer3', 'awayPlayer3_id',
       'awayPlayer4', 'awayPlayer4_id', 'awayPlayer5', 'awayPlayer5_id',
       'awayPlayer6', 'awayPlayer6_id', 'homePlayer1', 'homePlayer1_id',
       'homePlayer2', 'homePlayer2_id', 'homePlayer3', 'homePlayer3_id',
       'homePlayer4', 'homePlayer4_id', 'homePlayer5', 'homePlayer5_id',
       'homePlayer6', 'homePlayer6_id', 'Away_Players', 'Home_Players',
       'Away_Score', 'Home_Score', 'Away_Goalie', 'Away_Goalie_Id',
       'Home_Goalie', 'Home_Goalie_Id', 'xC', 'yC', 'Home_Coach',
       'Away_Coach'],

"""

if __name__ == "__main__":
    
    print("Hello World")
    # Read in the data
#     path = r"C:\Users\JoshG\hockey_scraper_data\csvs\nhl_pbp_20222023.csv"
    path = r"C:\Users\JoshG\hockey_scraper_data\csvs\nhl_pbp_20232024.csv"
    df = pd.read_csv(path)

    # print the column names of df
    column_names = df.columns

    # add a "prev_" prefix to every string in column_names
    prev_column_names = ['prev_' + name for name in column_names]

    # get list of ilocs of rows where the 'Event' column is 'GOAL or SHOT'
    # goal_shot_rows = df.loc[df['Event'].isin(['GOAL', 'SHOT'])].index.tolist()

    goal_shot_rows = df.loc[df['Event'].isin(['GOAL', 'SHOT'])]

    shifted_df = df.shift(1)
    shifted_df.columns = prev_column_names

    # combine goal_shot_rows with shifted_df, but using rows from goal_shot_rows
    goal_shot_rows = pd.concat([goal_shot_rows, shifted_df.loc[goal_shot_rows.index]], axis=1)


    print(goal_shot_rows["Event"])


    # save the new dataframe to a csv
    goal_shot_rows.to_csv("2023-2024_shots.csv", index=False)

    # create a subset of the dataframe, with only the 'Period', 'Seconds_Elapsed', 'Strength', 'Type', 'xC', 'yC'
    # 'prev_Event', 'prev_Period', 'prev_Seconds_Elapsed', 'prev_Strength', 'prev_Type', 'prev_xC', 'prev_yC'




"""
Index(['Unnamed: 0', 'Game_Id', 'Date', 'Period', 'Event', 'Description',
       'Time_Elapsed', 'Seconds_Elapsed', 'Strength', 'Ev_Zone', 'Type',
       'Ev_Team', 'Home_Zone', 'Away_Team', 'Home_Team', 'p1_name', 'p1_ID',
       'p2_name', 'p2_ID', 'p3_name', 'p3_ID', 'awayPlayer1', 'awayPlayer1_id',
       'awayPlayer2', 'awayPlayer2_id', 'awayPlayer3', 'awayPlayer3_id',
       'awayPlayer4', 'awayPlayer4_id', 'awayPlayer5', 'awayPlayer5_id',
       'awayPlayer6', 'awayPlayer6_id', 'homePlayer1', 'homePlayer1_id',
       'homePlayer2', 'homePlayer2_id', 'homePlayer3', 'homePlayer3_id',
       'homePlayer4', 'homePlayer4_id', 'homePlayer5', 'homePlayer5_id',
       'homePlayer6', 'homePlayer6_id', 'Away_Players', 'Home_Players',
       'Away_Score', 'Home_Score', 'Away_Goalie', 'Away_Goalie_Id',
       'Home_Goalie', 'Home_Goalie_Id', 'xC', 'yC', 'Home_Coach',
       'Away_Coach'],

"""