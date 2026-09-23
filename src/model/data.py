import pandas as pd
from src.classes.team import Team
from src.classes.player import Player
from src.model.calcweights import calcWeightsTeam
from src.model.calcweights import calcWeightsPlayer
def readAndClean(filepath):
    # drop duplicates and NA if filepath is found
    try:
        df = pd.read_csv(filepath)
        df = df.drop_duplicates().dropna()
        return df
    except FileNotFoundError:
        print("File is not found!")
        return
def teamsFilter(df):
    # filter columns
    df = df[df["situation"] == "all"]
    # create per game columns for the features we are looking for
    df["XGD"] = df["xGoalsFor"] / df["games_played"] - df["xGoalsAgainst"] / df["games_played"]
    df["GiveawayDifferential"] = df["giveawaysAgainst"] / df["games_played"] - df["giveawaysFor"] / df["games_played"]
    df["HDSD"] = df["highDangerShotsFor"] / df["games_played"] - df["highDangerShotsAgainst"] / df["games_played"]
    df["AGD"] = df["goalsFor"] / df["games_played"] - df["goalsAgainst"] / df["games_played"]
    # get the best model with its scaler and features
    model, scaler, featureNames = calcWeightsTeam()
    # get a list of all the teams in the file
    # create Team objects for all the teams and get all the feature stats for those teams as well
    teams = []
    for i in range(len(df)):
            row = df.iloc[i]
            teamName = row["name"]
            stats = {}
            for feature in featureNames:
                stats[feature] = row[feature]
            teams.append(Team(teamName, stats)) 
    # return the list, best model, and its scaler and features
    season = df["season"].iloc[0]
    return teams, model, scaler, featureNames, season
def playersFilter(df):
    # filter columns
    df = df[df["situation"] == "all"]
    # create per game columns for all stats which require them for accuracy
    perGameColumns = [
        "OnIce_F_goals",
        "OnIce_A_goals",
        "OnIce_F_xGoals",
        "OnIce_A_xGoals",
        "OnIce_F_shotsOnGoal",
        "OnIce_A_shotsOnGoal",
        "OnIce_F_shotAttempts",
        "OnIce_A_shotAttempts",
        "OnIce_F_unblockedShotAttempts",
        "OnIce_A_unblockedShotAttempts",
        "OnIce_F_highDangerShots",
        "OnIce_A_highDangerShots",
        "I_F_goals",
        "I_F_primaryAssists",
        "I_F_secondaryAssists",
        "I_F_points",
        "I_F_shotsOnGoal",
        "I_F_shotAttempts",
        "I_F_unblockedShotAttempts",
        "I_F_missedShots",
        "I_F_blockedShotAttempts",
        "I_F_takeaways",
        "I_F_giveaways",
        "I_F_hits",
        "I_F_rebounds",
        "I_F_reboundGoals",
        "I_F_savedShotsOnGoal"
    ]
    for column in perGameColumns:
        if column in df.columns:
            df[column] = df[column] / df["games_played"]
    # get the best model as well as its scaler and features
    model, scaler, featureNames = calcWeightsPlayer()
    # create a list of Player objects and their respective feature stats
    players = []
    for i in range(len(df)):
        row = df.iloc[i]
        playerName = row["name"]
        stats = {}
        for feature in featureNames:
            stats[feature] = row[feature]
        team = row["team"]
        players.append(Player(playerName, stats, team))
    # return the list, best model, and its scaler and features
    season = df["season"].iloc[0]
    return players, model, scaler, featureNames, season