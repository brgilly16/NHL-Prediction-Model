import os
import pandas as pd
from src.model.dailypower import calculateDailyPower
# Features will eventually include: Team PowerScore, Team Injury representation, Team Goaltending, HomeIce
# Target will be Team Goals For.
def loadDailyPower():
    # ensure the daily PowerScores are only calculated once and stored in a csv
    if not os.path.exists("data/daily_power_2025_26.csv"):
        calculateDailyPower()
    return pd.read_csv("data/daily_power_2025_26.csv")
def getFeatures(team, game, dailyPower, gameData):
    # the team's PowerScore going into this game, not its final season PowerScore
    df = dailyPower[(dailyPower["game"] == game) & (dailyPower["team"] == team)]
    teamScore = df["score"].iloc[0]
    gameData = gameData[(gameData["game"] == game) & (gameData["team"] == team)]
    missingPower = gameData["missingPower"].iloc[0]
    return teamScore, missingPower
