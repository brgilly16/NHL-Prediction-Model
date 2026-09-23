import os
import pandas as pd
from src.downloads.downloadteamgames import downloadTeamGames
from src.model.calcweights import calcWeightsTeam
def calculateDailyPower():
    # ensure that downloadTeamGames() is only called once and its data is stored in a csv
    if not os.path.exists("data/team_games_2025_26.csv"):
        downloadTeamGames()
    df = pd.read_csv("data/team_games_2025_26.csv")
    df = df.sort_values(["team", "gameDate", "gameId"])
    # running totals of each stat going INTO each game (shift(1) excludes the game itself)
    # so a team's score for a game only uses information available before puck drop
    stats = ["xGoalsFor", "xGoalsAgainst", "giveawaysFor", "giveawaysAgainst",
             "highDangerShotsFor", "highDangerShotsAgainst", "goalsFor", "goalsAgainst"]
    totals = df.groupby("team")[stats].transform(lambda s: s.cumsum().shift(1))
    gamesPlayed = df.groupby("team").cumcount()
    # build the same per game features the season level team model uses
    features = pd.DataFrame({
        "XGD": (totals["xGoalsFor"] - totals["xGoalsAgainst"]) / gamesPlayed,
        "xGoalsPercentage": totals["xGoalsFor"] / (totals["xGoalsFor"] + totals["xGoalsAgainst"]),
        "GiveawayDifferential": (totals["giveawaysAgainst"] - totals["giveawaysFor"]) / gamesPlayed,
        "HDSD": (totals["highDangerShotsFor"] - totals["highDangerShotsAgainst"]) / gamesPlayed,
        "AGD": (totals["goalsFor"] - totals["goalsAgainst"]) / gamesPlayed
    })
    daily = pd.DataFrame({
        "game": df["gameId"],
        "date": pd.to_datetime(df["gameDate"].astype(str)).dt.strftime("%Y-%m-%d"),
        "team": df["team"],
        "gamesPlayed": gamesPlayed
    })
    # a team's first game has no prior data, so it gets no score
    hasData = gamesPlayed > 0
    model, scaler, featureNames = calcWeightsTeam()
    daily["score"] = float("nan")
    daily.loc[hasData, "score"] = model.predict(scaler.transform(features.loc[hasData, featureNames]))
    daily.to_csv("data/daily_power_2025_26.csv", index=False)
    print("Finished! Rows:", len(daily))
    return daily
if __name__ == "__main__":
    calculateDailyPower()
