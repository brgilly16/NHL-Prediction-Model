# ALL WRITTEN BY AI
import pandas as pd
import requests
from io import StringIO
import time


def downloadParticipation():

    baseURL = "https://moneypuck.com/moneypuck/playerData/playerGameByGame/2025/regular/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(baseURL, headers=headers)

    if response.status_code != 200:
        print("Could not access MoneyPuck.")
        print("Status:", response.status_code)
        return

    lines = response.text.splitlines()

    files = []

    for line in lines:

        if ".csv" in line:

            start = line.find('href="') + 6
            end = line.find('"', start)

            if start > 5 and end > start:
                filename = line[start:end]

                if filename.endswith(".csv"):
                    files.append(filename)

    print("Player files found:", len(files))

    participation = []

    for filename in files:

        print("Downloading", filename)

        url = baseURL + filename

        try:

            response = requests.get(
                url,
                headers=headers
            )

            if response.status_code != 200:
                print("Status:", response.status_code)
                continue

            df = pd.read_csv(
                StringIO(response.text)
            )

            if df.empty:
                continue

            columns = [
            "playerId",
            "name",
            "gameId",
            "playerTeam",
            "gameDate",
            "position"
            ]

            columns = [
                column
                for column in columns
                if column in df.columns
            ]

            df = df[columns]

            participation.append(df)

        except Exception as e:

            print("Error:", e)

        time.sleep(0.5)

    if len(participation) == 0:

        print("No player data downloaded.")
        return

    participation = pd.concat(
        participation,
        ignore_index=True
    )

    participation = participation.drop_duplicates()

    participation = participation.rename(columns={
    "playerId": "playerID",
    "gameId": "game",
    "gameDate": "date",
    "playerTeam": "team",
    "position": "position"
    })

    participation["played"] = 1

    participation.to_csv(
        "data/player_participation_2025_26.csv",
        index=False
    )

    print()
    print("Finished!")
    print("Rows:", len(participation))
if __name__ == "__main__":
    downloadParticipation()