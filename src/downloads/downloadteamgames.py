import pandas as pd
import requests
from io import StringIO
import time
def downloadTeamGames():
    # MoneyPuck stores one game-by-game csv per team for the season
    baseURL = "https://moneypuck.com/moneypuck/playerData/teamGameByGame/2025/regular/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(baseURL, headers=headers)
    if response.status_code != 200:
        print("Could not access MoneyPuck. Status:", response.status_code)
        return
    # get every team csv listed on the index page
    files = []
    for line in response.text.splitlines():
        start = line.find('href="') + 6
        end = line.find('"', start)
        if start > 5 and end > start and line[start:end].endswith(".csv"):
            files.append(line[start:end])
    teamGames = []
    for filename in files:
        print("Downloading", filename)
        response = requests.get(baseURL + filename, headers=headers)
        if response.status_code != 200:
            print("Status:", response.status_code)
            continue
        df = pd.read_csv(StringIO(response.text))
        # only keep the all situations rows, matching the season level team model
        teamGames.append(df[df["situation"] == "all"])
        time.sleep(0.5)
    if len(teamGames) == 0:
        print("No team data downloaded.")
        return
    teamGames = pd.concat(teamGames, ignore_index=True).drop_duplicates()
    teamGames.to_csv("data/team_games_2025_26.csv", index=False)
    print("Finished! Rows:", len(teamGames))
if __name__ == "__main__":
    downloadTeamGames()
