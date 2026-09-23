import os
import pandas as pd
from src.classes.player import Player
def getTeams(players):
    teams = []
    for player in players:
        teams.append(player.team)
    return teams
def storeRankings(items, scores, season,category, time):
    if isinstance(items[0], Player):
        teams = getTeams(items)
    else:
        teams = [""] * len(items)
    newRankings = pd.DataFrame({
        "name": items,
        "score": scores,
        "season": [season] * len(items),
        "category": [category] * len(items),
        "time": [time] * len(items),
        "playerTeam": teams
        })
    if not os.path.exists("data/rankings.csv"):
        newRankings.to_csv("data/rankings.csv", index=False)
        return
    df = pd.read_csv("data/rankings.csv")
    if ((df["season"] == season) & (df["category"] == category) & (df["time"] == time)).any():
        return
    df = pd.concat([df, newRankings], ignore_index=True)
    df.to_csv("data/rankings.csv", index=False)