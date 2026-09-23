import pandas as pd
import os
from src.downloads.downloadgar import downloadGar
from src.downloads.downloadstandings import downloadStandings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
def calcWeightsTeam():
    # always use regular season data for the model
    df = pd.read_csv("data/teams_2008_to_2024.csv")
    # filter df and drop duplicates
    df = df.drop_duplicates()
    df = df[df["situation"] == "all"]
    # create per game columns for stats that require them
    df["XGD"] = df["xGoalsFor"] / df["games_played"] - df["xGoalsAgainst"] / df["games_played"]
    df["GiveawayDifferential"] = df["giveawaysAgainst"] / df["games_played"] - df["giveawaysFor"] / df["games_played"]
    df["HDSD"] = df["highDangerShotsFor"] / df["games_played"] - df["highDangerShotsAgainst"] / df["games_played"]
    df["AGD"] = df["goalsFor"] / df["games_played"] - df["goalsAgainst"] / df["games_played"]
    # fix mismatched team abbreviations with hockey reference data
    teamMap = {
        "S.J": "SJS",
        "N.J": "NJD",
        "T.B": "TBL",
        "L.A": "LAK"
    }
    df["team"] = df["team"].replace(teamMap)
    # get PTS% column and add it to df through the merge
    # ensure that it only calls downloadStandings() once (in the case that the csv filepath does not exist) and stores the data in a csv with the rest of the project data
    if not os.path.exists("data/standings.csv"):
        downloadStandings()
    standings = pd.read_csv("data/standings.csv")
    df = df.merge(standings, on = ["team", "season"], how = "left")
    # set features and target
    X = df[["XGD", "xGoalsPercentage", "GiveawayDifferential", "HDSD", "AGD"]]
    Y = df["PTS%"]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # use both linear and ridge regression models and determine which is best via their r2 scores
    # choose the best r2 score out of the two
    modelOne = LinearRegression()
    modelTwo = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    modelOne.fit(X_train_scaled, Y_train)
    modelTwo.fit(X_train_scaled, Y_train)
    predictionsOne = modelOne.predict(X_test_scaled)
    predictionsTwo = modelTwo.predict(X_test_scaled)
    accOne = r2_score(Y_test, predictionsOne)
    accTwo = r2_score(Y_test, predictionsTwo)
    if accOne > accTwo:
        bestModel = modelOne
        print("modelOne chosen")
    else:
        bestModel = modelTwo
        print("modelTwo chosen")
    print(accOne, accTwo)
    featureNames = X.columns.tolist()
    # return the best model, its scaler, and its features
    return bestModel, scaler, featureNames
def calcWeightsPlayer():
    df = pd.read_csv("data/skaters (1).csv")
    # drop duplicates and NA
    df = df.drop_duplicates().dropna()
    # filter df
    df = df[df["situation"] == "all"]
    # create per game columns for the stats which require them
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
    # get the GAR stat and add it to df via a left join merge
    # enusre that downloadGar() is only called once that way the data can be stored with the rest of the data for the project
    if not os.path.exists("data/gar.csv"):
        downloadGar()
    table = pd.read_csv("data/gar.csv")
    df = df.merge(table, left_on = ["season", "name"], right_on = ["Season","Player"], how = "left")
    # drop any NA in the GAR column
    df = df.dropna(subset=["GAR"])
    # filter for players who have played more than 20 games
    df = df[df["games_played"] > 20]
    # exclude a number of stats which could leak information from the target
    # exclude stats that are not numerical or are not useful for predicting GAR
    exclude = [
    "GAR",
    "WAR",
    "SPAR",
    "EVO_GAR",
    "EVD_GAR",
    "PPO_GAR",
    "SHD_GAR",
    "Take_GAR",
    "Draw_GAR",
    "Off_GAR",
    "Def_GAR",
    "Pens_GAR",
    "playerId",
    "name",
    "games_played",
    "icetime",
    "iceTimeRank",
    "shifts",
    "timeOnBench",
    "Player",
    "team",
    "Team",
    "season",
    "Season",
    "position",
    "Position",
    "situation",
    "GP",
    "TOI_All",
    "I_F_shifts",
    "faceoffsWon",
    "penalityMinutes",
    "gameScore",
    "xGoalsForAfterShifts",
    "xGoalsAgainstAfterShifts",
    "corsiForAfterShifts",
    "corsiAgainstAfterShifts",
    "fenwickForAfterShifts",
    "fenwickAgainstAfterShifts"
    ]
    # set features and target
    X = df.drop(columns=exclude)
    X = X.select_dtypes(include="number")   
    Y = df["GAR"]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # train three separate model types and determine which is best via r2 score
    modelOne = LinearRegression()
    modelTwo = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    modelThree = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, max_iter=1000000, random_state=42)
    modelOne.fit(X_train_scaled, Y_train)
    modelTwo.fit(X_train_scaled, Y_train)
    modelThree.fit(X_train_scaled, Y_train)
    predictionsOne = modelOne.predict(X_test_scaled)
    predictionsTwo = modelTwo.predict(X_test_scaled)
    predictionsThree = modelThree.predict(X_test_scaled)
    accOne = r2_score(Y_test, predictionsOne)
    accTwo = r2_score(Y_test, predictionsTwo)
    accThree = r2_score(Y_test, predictionsThree)
    # select the best model
    if accOne >= accTwo and accOne >= accThree:
        bestModel = modelOne
        print("modelOne chosen")
    elif accTwo >= accOne and accTwo >= accThree:
        bestModel = modelTwo
        print("modelTwo chosen")
    else:
        bestModel = modelThree
        print("modelThree chosen")
    print(accOne, accTwo, accThree)
    featureNames = X.columns.tolist()
    # return the best model, its scaler, and its features
    return bestModel, scaler, featureNames