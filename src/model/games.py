# ALL WRITTEN BY AI
import pandas as pd
import requests


def createGameData():

    missingPower = pd.read_csv(
        "data/missing_power_2025_26.csv"
    )

    games = missingPower[
        ["game", "date", "team", "missingPower"]
    ].copy()

    # Get NHL game results
    results = []

    for offset in range(0, 1400, 100):

        url = (
            "https://project94hockey.com/v1/games"
            f"?season=20252026&limit=100&offset={offset}"
        )

        response = requests.get(url)

        if response.status_code != 200:
            print("Could not download game results.")
            print("Status:", response.status_code)
            break

        data = response.json()

        if len(data["data"]) == 0:
            break

        results.extend(data["data"])

    results = pd.DataFrame(results)

    results = results[
        [
            "game_id",
            "game_date",
            "home_team_abbrev",
            "home_score",
            "away_team_abbrev",
            "away_score"
        ]
    ]

    # Convert home and away results into the same
    # one-team-per-row format as game_data.csv

    home = results[
        [
            "game_id",
            "game_date",
            "home_team_abbrev",
            "home_score"
        ]
    ].copy()

    home = home.rename(columns={
        "game_id": "game",
        "game_date": "date",
        "home_team_abbrev": "team",
        "home_score": "goals"
    })

    away = results[
        [
            "game_id",
            "game_date",
            "away_team_abbrev",
            "away_score"
        ]
    ].copy()

    away = away.rename(columns={
        "game_id": "game",
        "game_date": "date",
        "away_team_abbrev": "team",
        "away_score": "goals"
    })

    scores = pd.concat(
        [home, away],
        ignore_index=True
    )

    # Make sure game IDs have the same type
    games["game"] = games["game"].astype(str)
    scores["game"] = scores["game"].astype(str)

    # Add goals to game_data
    games = games.merge(
        scores[["game", "team", "goals"]],
        on=["game", "team"],
        how="left"
    )

    games = games.sort_values(
        ["game", "date", "team"]
    )

    games.to_csv(
        "data/game_data.csv",
        index=False
    )

    print()
    print("Finished!")
    print("Games:", games["game"].nunique())
    print("Rows:", len(games))
    print("Games without goals:", games["goals"].isna().sum())

    return games