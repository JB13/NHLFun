import pandas as pd
import numpy as np


def convert_odds_to_df(path_to_csv):
    # read in "NHL_odds_2021-2022.csv"
    df = pd.read_csv(path_to_csv)

    print(df.columns)

    # make a new column based off of open column where if it's positive, it's x/(x+100) and if it's negative, it's x/(x-100)
    df['open_prob'] = np.where(df['Open'] > 0, (1- (df['Open'] / (df['Open'] + 100))), df['Open'] / (df['Open'] - 100))

    

    return df


if __name__ == "__main__":
    # Read in the data
    path = "NHL_odds_2021-2022.csv"
    df = convert_odds_to_df(path)

    # split up the dfs into multiple dfs based off of the team
    mult_dfs = [df.loc[df['Team'] == team] for team in df['Team'].unique()]

    print(df.head())

    print(mult_dfs[0].head())

    #