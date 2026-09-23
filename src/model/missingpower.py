# ALL WRITTEN BY AI
import pandas as pd


def calculateMissingPower():

    participation = pd.read_csv(
        "data/player_participation_2025_26.csv"
    )

    rankings = pd.read_csv(
        "data/rankings.csv"
    )

    rankings = rankings[
        (rankings["category"] == "players") &
        (rankings["season"] == 2025) &
        (rankings["time"] == "regular")
    ]

    rankings = rankings[
        ["name", "score"]
    ].drop_duplicates(
        subset=["name"]
    )

    # Convert dates
    participation["date"] = pd.to_datetime(
        participation["date"].astype(str)
    )

    # -------------------------------------------------
    # Find each player's first and last game
    # -------------------------------------------------

    playerRanges = participation.groupby(
        ["playerID", "name", "team"]
    )["date"].agg(
        firstGame="min",
        lastGame="max"
    ).reset_index()

    # -------------------------------------------------
    # Get every game each team played
    # -------------------------------------------------

    games = participation[
        ["game", "date", "team"]
    ].drop_duplicates()

    games = games.merge(
        playerRanges,
        on="team",
        how="left"
    )

    games = games[
        (games["date"] >= games["firstGame"]) &
        (games["date"] <= games["lastGame"])
    ]

    # -------------------------------------------------
    # Determine whether each player actually played
    # -------------------------------------------------

    played = participation[
        ["game", "team", "playerID"]
    ].drop_duplicates()

    played["played"] = 1

    games = games.merge(
        played,
        on=["game", "team", "playerID"],
        how="left"
    )

    games["played"] = games["played"].fillna(0)

    # -------------------------------------------------
    # Players who were available but did not play
    # -------------------------------------------------

    missing = games[
        games["played"] == 0
    ].copy()

    # -------------------------------------------------
    # Add Power Score
    # -------------------------------------------------

    missing = missing.merge(
        rankings,
        on="name",
        how="left"
    )

    # Missing Power should only count POSITIVE value
    missing["score"] = missing["score"].fillna(0)
    missing["score"] = missing["score"].clip(lower=0)

    # -------------------------------------------------
    # REMOVE GOALIES
    # -------------------------------------------------

    # Get player positions from the participation data
    positions = participation[
        ["playerID", "name"]
    ].drop_duplicates()

    # MoneyPuck uses position information in the original
    # player files, but it is not currently saved in our
    # participation CSV. Therefore, this section will only
    # work if "position" exists in the participation file.

    if "position" in missing.columns:
        missing = missing[
            ~missing["position"].isin(["G", "Goalie"])
        ]

    # -------------------------------------------------
    # Calculate total missing Power
    # -------------------------------------------------

    missingPower = missing.groupby(
        ["game", "date", "team"]
    ).agg(
        missingPower=("score", "sum")
    ).reset_index()

    # -------------------------------------------------
    # List missing players
    # -------------------------------------------------

    missingPlayers = missing.groupby(
        ["game", "date", "team"]
    )["name"].apply(
        lambda x: ", ".join(x)
    ).reset_index(
        name="missingPlayers"
    )

    missingPower = missingPower.merge(
        missingPlayers,
        on=["game", "date", "team"],
        how="left"
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    missingPower.to_csv(
        "data/missing_power_2025_26.csv",
        index=False
    )

    print("Finished!")
    print(
        "Games:",
        missingPower["game"].nunique()
    )

    print(
        "Rows:",
        len(missingPower)
    )

    return missingPower
if __name__ == "__main__":
    calculateMissingPower()