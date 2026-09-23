import os
import re
import json
import time
import datetime
import requests
import pandas as pd
from io import StringIO
# MoneyPuck game-by-game data used by the game prediction model
# history: one-time career downloads (every season up to when they were downloaded)
# seasons: the current season (and any season after the history) kept up to date file by file,
#          only re-downloading files MoneyPuck changed since the last run
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/"
SEASON_BASE = "https://moneypuck.com/moneypuck/playerData/"
DATA = "data/gamemodel/"
GOALIE_COLUMNS = ["playerId", "season", "name", "gameId", "playerTeam", "gameDate", "icetime", "xGoals", "goals", "ongoal"]
def get(url):
    # a few retries so one network hiccup does not fail a daily run
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            if response.status_code in (200, 404):
                return response
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return response
def listing(url):
    # csv files in a MoneyPuck folder with their last modified time
    response = get(url)
    if response.status_code != 200:
        return {}
    return dict(re.findall(r'href="([^"/]+\.csv)".*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', response.text))
def currentSeason(today=None):
    # NHL seasons are named by the year they start (2026 = 2026-27); a new season counts from September
    today = today or datetime.date.today()
    return today.year if today.month >= 9 else today.year - 1
def downloadAllTeams():
    # every team's game-by-game stats since 2008 (regular season and playoffs) in a single file
    os.makedirs(DATA, exist_ok=True)
    response = get(BASE + "all_teams.csv")
    if response.status_code != 200:
        print("Could not download team games. Status:", response.status_code)
        return
    with open(DATA + "all_teams_raw.csv", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Finished downloading team games")
def downloadGoalies():
    # one game-by-game csv per goalie, only the all situations rows are kept
    os.makedirs(DATA, exist_ok=True)
    files = list(listing(BASE + "regular/goalies/"))
    goalies = []
    for i, filename in enumerate(files):
        response = get(BASE + "regular/goalies/" + filename)
        if response.status_code != 200:
            print(filename, "status:", response.status_code)
            continue
        df = pd.read_csv(StringIO(response.text))
        goalies.append(df[df["situation"] == "all"][GOALIE_COLUMNS])
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
    files = list(listing(BASE + "regular/skaters/"))
    batchSize = 250
    for batch in range(0, len(files), batchSize):
        path = folder + f"part_{batch // batchSize:03d}.parquet"
        if os.path.exists(path):
            continue
        skaters = []
        for filename in files[batch:batch + batchSize]:
            response = get(BASE + "regular/skaters/" + filename)
            if response.status_code != 200:
                print(filename, "status:", response.status_code)
                continue
            df = pd.read_csv(StringIO(response.text))
            skaters.append(df[df["situation"] == "all"].drop(columns=["situation"]))
            time.sleep(0.1)
        pd.concat(skaters, ignore_index=True).to_parquet(path, index=False)
        print("Saved skaters", min(batch + batchSize, len(files)), "of", len(files))
    print("Finished downloading skater games")
def historySeason():
    # the last season in the career history download
    seasons = pd.read_csv(DATA + "all_teams_raw.csv", usecols=["season"])["season"]
    return int(seasons.max())
def syncedSeasons():
    # seasons that come from the per-season folders instead of the history (always includes the current season)
    current = currentSeason()
    return list(range(min(historySeason() + 1, current), current + 1))
def syncFolder(url, folder, keepAllSituations):
    # download only the files MoneyPuck changed since the last sync
    os.makedirs(folder, exist_ok=True)
    manifestPath = folder + "manifest.json"
    manifest = json.load(open(manifestPath)) if os.path.exists(manifestPath) else {}
    files = listing(url)
    changed = [f for f, stamp in files.items() if manifest.get(f) != stamp]
    for i, filename in enumerate(changed):
        response = get(url + filename)
        if response.status_code != 200:
            print(filename, "status:", response.status_code)
            continue
        df = pd.read_csv(StringIO(response.text))
        if not keepAllSituations:
            df = df[df["situation"] == "all"]
        df.to_csv(folder + filename, index=False)
        manifest[filename] = files[filename]
        if i % 100 == 99:
            json.dump(manifest, open(manifestPath, "w"))
        time.sleep(0.1)
    json.dump(manifest, open(manifestPath, "w"))
    return len(changed)
def syncSeasons():
    for season in syncedSeasons():
        folder = DATA + f"seasons/{season}/"
        teams = syncFolder(SEASON_BASE + f"teamGameByGame/{season}/regular/", folder + "teams/", True)
        players = syncFolder(SEASON_BASE + f"playerGameByGame/{season}/regular/", folder + "players/", False)
        print(f"Season {season}: updated {teams} team files and {players} player files")
def seasonFiles(season, kind):
    folder = DATA + f"seasons/{season}/{kind}/"
    if not os.path.isdir(folder):
        return pd.DataFrame()
    frames = [pd.read_csv(folder + f) for f in sorted(os.listdir(folder)) if f.endswith(".csv")]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True).assign(season=season) if frames else pd.DataFrame()
def loadTeamRaw(columns):
    # team game-by-game rows: history for older seasons, synced files for the synced seasons
    synced = syncedSeasons()
    history = pd.read_csv(DATA + "all_teams_raw.csv", usecols=columns)
    frames = [history[~history["season"].isin(synced)]]
    for season in synced:
        current = seasonFiles(season, "teams")
        if len(current):
            current["playoffGame"] = 0
            frames.append(current[columns])
    return pd.concat(frames, ignore_index=True)
def loadGoalieGames():
    synced = syncedSeasons()
    history = pd.read_csv(DATA + "goalie_games.csv")
    frames = [history[~history["season"].isin(synced)]]
    for season in synced:
        current = seasonFiles(season, "players")
        if len(current):
            frames.append(current[current["position"] == "G"][GOALIE_COLUMNS])
    return pd.concat(frames, ignore_index=True)
def loadSkaters():
    synced = syncedSeasons()
    folder = DATA + "skaters/"
    parts = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    history = pd.concat([pd.read_parquet(folder + f) for f in parts], ignore_index=True)
    frames = [history[~history["season"].isin(synced)]]
    for season in synced:
        current = seasonFiles(season, "players")
        if len(current):
            frames.append(current[current["position"] != "G"].drop(columns=["situation"]))
    return pd.concat(frames, ignore_index=True)
def fetchRosters(teams):
    # current NHL rosters (the NHL's public API), so offseason trades, signings and call-ups show up before a player's first game
    # player ids are the same ones MoneyPuck uses; a team whose roster cannot be fetched is left out (its last known lineup is used)
    rows = []
    for team in teams:
        response = get(f"https://api-web.nhle.com/v1/roster/{team}/current")
        if response.status_code != 200:
            print("Could not fetch roster for", team)
            continue
        for group, players in response.json().items():
            for p in players:
                rows.append({"playerId": p["id"], "name": f'{p["firstName"]["default"]} {p["lastName"]["default"]}',
                             "team": team, "position": p.get("positionCode")})
        time.sleep(0.2)
    rosters = pd.DataFrame(rows, columns=["playerId", "name", "team", "position"])
    rosters.to_csv(DATA + "rosters.csv", index=False)
    print("Fetched rosters:", rosters["team"].nunique(), "teams,", len(rosters), "players")
    return rosters
def downloadAll():
    if not os.path.exists(DATA + "all_teams_raw.csv"):
        downloadAllTeams()
    if not os.path.exists(DATA + "goalie_games.csv"):
        downloadGoalies()
    if not os.path.exists(DATA + "skaters/done"):
        downloadSkaters()
        open(DATA + "skaters/done", "w").close()
    syncSeasons()
if __name__ == "__main__":
    downloadAll()
