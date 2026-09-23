import os
import time
import requests
import pandas as pd
from io import StringIO
# MoneyPuck game-by-game data used by the game prediction model
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/"
DATA = "data/gamemodel/"
def downloadAllTeams():
    # every team's game-by-game stats since 2008 (regular season and playoffs) in a single file
    os.makedirs(DATA, exist_ok=True)
    response = requests.get(BASE + "all_teams.csv", headers=HEADERS)
    if response.status_code != 200:
        print("Could not download team games. Status:", response.status_code)
        return
    with open(DATA + "all_teams_raw.csv", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Finished downloading team games")
def downloadGoalies():
    # one game-by-game csv per goalie, only the all situations rows are kept
    os.makedirs(DATA, exist_ok=True)
    response = requests.get(BASE + "regular/goalies/", headers=HEADERS)
    if response.status_code != 200:
        print("Could not access MoneyPuck. Status:", response.status_code)
        return
    files = []
    for line in response.text.splitlines():
        start = line.find('href="') + 6
        end = line.find('"', start)
        if start > 5 and end > start and line[start:end].endswith(".csv"):
            files.append(line[start:end])
    columns = ["playerId", "season", "name", "gameId", "playerTeam", "gameDate", "icetime", "xGoals", "goals", "ongoal"]
    goalies = []
    for i, filename in enumerate(files):
        response = requests.get(BASE + "regular/goalies/" + filename, headers=HEADERS)
        if response.status_code != 200:
            print(filename, "status:", response.status_code)
            continue
        df = pd.read_csv(StringIO(response.text))
        goalies.append(df[df["situation"] == "all"][columns])
        if i % 25 == 0:
            print("Downloaded", i + 1, "of", len(files), "goalies")
        time.sleep(0.3)
    pd.concat(goalies, ignore_index=True).to_csv(DATA + "goalie_games.csv", index=False)
    print("Finished downloading goalie games")
def downloadSkaters():
    # one game-by-game csv per skater since 2008, only the all situations rows are kept
    # saved in batches so an interrupted download can resume where it stopped
    folder = DATA + "skaters/"
    os.makedirs(folder, exist_ok=True)
    response = requests.get(BASE + "regular/skaters/", headers=HEADERS)
    if response.status_code != 200:
        print("Could not access MoneyPuck. Status:", response.status_code)
        return
    files = []
    for line in response.text.splitlines():
        start = line.find('href="') + 6
        end = line.find('"', start)
        if start > 5 and end > start and line[start:end].endswith(".csv"):
            files.append(line[start:end])
    batchSize = 250
    for batch in range(0, len(files), batchSize):
        path = folder + f"part_{batch // batchSize:03d}.parquet"
        if os.path.exists(path):
            continue
        skaters = []
        for filename in files[batch:batch + batchSize]:
            for attempt in range(3):
                try:
                    response = requests.get(BASE + "regular/skaters/" + filename, headers=HEADERS, timeout=60)
                    break
                except requests.RequestException:
                    time.sleep(5)
            if response.status_code != 200:
                print(filename, "status:", response.status_code)
                continue
            df = pd.read_csv(StringIO(response.text))
            skaters.append(df[df["situation"] == "all"].drop(columns=["situation"]))
            time.sleep(0.1)
        pd.concat(skaters, ignore_index=True).to_parquet(path, index=False)
        print("Saved skaters", min(batch + batchSize, len(files)), "of", len(files))
    print("Finished downloading skater games")
def loadSkaters():
    folder = DATA + "skaters/"
    parts = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    return pd.concat([pd.read_parquet(folder + f) for f in parts], ignore_index=True)
def downloadAll():
    if not os.path.exists(DATA + "all_teams_raw.csv"):
        downloadAllTeams()
    if not os.path.exists(DATA + "goalie_games.csv"):
        downloadGoalies()
    if not os.path.exists(DATA + "skaters/done"):
        downloadSkaters()
        open(DATA + "skaters/done", "w").close()
if __name__ == "__main__":
    downloadAll()
