import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
def create_input_data_from_csv(path_to_csv):
    df = pd.read_csv(path_to_csv)

    print(len(df))
    print("HELLO WORLD 2")
    
    # get list of ilocs of rows where the 'Event' column is 'GOAL or SHOT'
    goal_shot_rows = df.loc[df['Event'].isin(['GOAL', 'SHOT'])]

    print(goal_shot_rows["Event"])

    print(list(goal_shot_rows.columns.values).count("prev_xC"))

    # shift the dataframe by 1 row
    shifted_df = df.shift(1)
    shifted_df.columns = ['prev_' + name for name in df.columns]

    print(list(shifted_df.columns.values).count("prev_xC"))

    # combine goal_shot_rows with shifted_df, but using rows from goal_shot_rows
    goal_shot_rows = pd.concat([goal_shot_rows, shifted_df.loc[goal_shot_rows.index]], axis=1)
    print(list(goal_shot_rows.columns.values).count("prev_xC"))

    goal_shot_rows = goal_shot_rows[goal_shot_rows['Period'] != 0]
    goal_shot_rows = goal_shot_rows[goal_shot_rows['Period'] != 5]

    val_input_data = goal_shot_rows[['Event', 'Period', 'Seconds_Elapsed', 'Strength', 'Type', 'xC', 'yC', 'prev_Event', 'prev_Period', 'prev_Seconds_Elapsed', 'prev_Strength', 'prev_Type', 'prev_xC', 'prev_yC']]
    val_result_data = goal_shot_rows['Event']
    val_result_data = val_result_data.apply(lambda x: 1 if x == "GOAL" else 0)

    val_input_data = val_input_data.drop(columns=['Event'])
    print(list(val_input_data.columns.values).count("prev_xC"))

    # Split out the data between numeric values (can carry forward) and categorical values (need to be turned into binary columns)
    val_input_data_numeric = val_input_data[['Period', 'Seconds_Elapsed', 'xC', 'yC', 'prev_Seconds_Elapsed', 'prev_xC', 'prev_yC']]
    val_input_data_categorical = pd.DataFrame()

    for column in ['Strength', 'Type', 'prev_Event']:
        dummy_columns = pd.get_dummies(val_input_data[column])
        val_input_data_categorical = pd.concat([dummy_columns, val_input_data_categorical], axis=1)

    for column in ['prev_Event', 'prev_Strength', 'prev_Type']:
        dummy_columns = pd.get_dummies(val_input_data[column])

        # rename all columns with a prefix of "prev_"
        dummy_columns.columns = ['prev_' + str(col) for col in dummy_columns.columns]
        val_input_data_categorical = pd.concat([val_input_data_categorical, dummy_columns], axis=1)
        
    # change all True/False to 1/0
    val_input_data_categorical = val_input_data_categorical.applymap(lambda x: 1 if x == True else 0)

    # combine the two dataframes
    val_input_data_combined = pd.concat([val_input_data_numeric, val_input_data_categorical], axis=1)

    #set val_input_data2 to be all floats
    val_input_data_combined = val_input_data_combined.astype(float)

    columns = ['Period', 'Seconds_Elapsed', 'xC', 'yC', 'prev_Seconds_Elapsed',
            'prev_xC', 'prev_yC', 'BLOCK', 'CHL', 'DELPEN', 'FAC', 'GIVE', 'HIT',
            'MISS', 'PENL', 'SHOT', 'STOP', 'TAKE', 'BACKHAND', 'DEFLECTED',
            'SLAP SHOT', 'SNAP SHOT', 'TIP-IN', 'WRAP-AROUND', 'WRIST SHOT', '0x0',
            '3x3', '3x4', '3x5', '4x3', '4x4', '4x5', '5x3', '5x4', '5x5', '6x5',
            'prev_BLOCK', 'prev_CHL', 'prev_DELPEN', 'prev_FAC', 'prev_GIVE',
            'prev_HIT', 'prev_MISS', 'prev_PENL', 'prev_SHOT', 'prev_STOP',
            'prev_TAKE', 'prev_0x5', 'prev_3x3', 'prev_3x4', 'prev_3x5', 'prev_4x3',
            'prev_4x4', 'prev_4x5', 'prev_5x3', 'prev_5x4', 'prev_5x5', 'prev_5x6',
            'prev_BACKHAND', 'prev_DEFLECTED',
            'prev_PS-Covering puck in crease(0 min)',
            'prev_PS-Goalkeeper displaced net(0 min)',
            'prev_PS-Holding on breakaway(0 min)',
            'prev_PS-Hooking on breakaway(0 min)',
            'prev_PS-Slash on breakaway(0 min)',
            'prev_PS-Throw object at puck(0 min)',
            'prev_PS-Tripping on breakaway(0 min)', 'prev_SLAP SHOT',
            'prev_SNAP SHOT', 'prev_TIP-IN', 'prev_WRAP-AROUND', 'prev_WRIST SHOT']

    print(val_input_data_combined.shape)
    print(val_input_data_combined.columns)
    # For each column in input_data_combined, check if it exists in val_input_data_combined, if not add it as all 0s
    for column in columns:
        if column not in val_input_data_combined.columns:
                val_input_data_combined[column] = 0

    for column in val_input_data_combined.columns:
        if column not in columns:
                if column in val_input_data_combined.columns:
                    print(f"Column {column} not in columns")
                    val_input_data_combined = val_input_data_combined.drop(column, axis=1)

    print(val_input_data_combined.shape)
    print(val_input_data_combined.columns)

    # reduce val_input_data_combined to have the same columns as input_data_combined
    val_input_data_combined = val_input_data_combined[columns]

    # change all NaN numbers to 0
    val_input_data_combined = val_input_data_combined.fillna(0)

    print(val_input_data_combined.shape)

    filepath = '0422Model_800k.pt'
    model = nn.Sequential(
        nn.Linear(72, 120),
        nn.ReLU(),
        nn.Linear(120, 60),
        nn.ReLU(),
        nn.Linear(60, 1),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load(filepath))
    model.eval()
    model.cpu()

    x = torch.tensor(val_input_data_combined.values, dtype=torch.float32)
    y = torch.tensor(val_result_data.values, dtype=torch.float32)
    result = model(x)

    # add the probability of a goal to val_input_data_combined
    goal_shot_rows['Goal_Probability'] = result.detach().numpy()

    return val_input_data_combined, val_result_data, goal_shot_rows


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