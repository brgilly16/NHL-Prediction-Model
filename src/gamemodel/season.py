import os
import datetime
import numpy as np
import pandas as pd
from src.gamemodel.download import DATA, get
from src.gamemodel.predict import Predictor
from src.gamemodel.train import GameModel, loadFeatures, homeGames, awayRows
# the current season's live track record: the prediction the site published before each game, and how the game turned out
# the log is committed to the repo by the GitHub Action, so it builds up over the season
LOG = DATA + "season_predictions.csv"
COLUMNS = ["gameId", "season", "date", "start", "home", "away", "homeGoalie", "homeGoalieStatus", "awayGoalie", "awayGoalieStatus",
           "homeRest", "awayRest", "pHome", "homeRate", "awayRate", "predictedAt", "source",
           "homeScore", "awayScore", "decidedBy"]
def loadLog():
    return pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLUMNS)
def logPredictions(log, schedule, goalieNames, season):
    # every game that has not started gets its latest pregame prediction; a game's row is frozen once it starts
    now = datetime.datetime.now(datetime.timezone.utc)
    predictor = Predictor()
    started = set(log.loc[pd.to_datetime(log["start"], utc=True) <= now, "gameId"]) if len(log) else set()
    rows = {r["gameId"]: r for r in log.to_dict("records")}
    for g in schedule:
        if pd.Timestamp(g["start"]) <= now or g["id"] in started:
            continue
        r = predictor.predict(g["home"], g["away"], homeGoalie=g.get("homeGoalie"), awayGoalie=g.get("awayGoalie"),
                              homeRest=g["homeRest"], awayRest=g["awayRest"])
        rows[g["id"]] = {"gameId": g["id"], "season": season, "date": g["date"], "start": g["start"], "home": g["home"], "away": g["away"],
                         "homeGoalie": goalieNames.get((g["home"], g.get("homeGoalie")), r["homeGoalie"]), "homeGoalieStatus": g.get("homeGoalieStatus"),
                         "awayGoalie": goalieNames.get((g["away"], g.get("awayGoalie")), r["awayGoalie"]), "awayGoalieStatus": g.get("awayGoalieStatus"),
                         "homeRest": g["homeRest"], "awayRest": g["awayRest"], "pHome": r["homeWin"], "homeRate": r["homeGoals"],
                         "awayRate": r["awayGoals"], "predictedAt": now.strftime("%Y-%m-%dT%H:%MZ"), "source": "live"}
    return pd.DataFrame(list(rows.values()), columns=COLUMNS)
def backfill(log, season):
    # games played before logging started are predicted by the model as it stood before the season (trained only on earlier
    # seasons), from each game's pregame features, so they are still out of sample; they are marked "reconstructed"
    df = loadFeatures()
    games = df[df["season"] == season]
    games = games[~games["gameId"].isin(log["gameId"])]
    if not len(games) or not (df["season"] < season).any():
        return log
    import json
    best = json.load(open(DATA + "model_report.json"))["best"]
    model = GameModel(**best).fit(df[df["season"] < season])
    predictions = model.predictGames(games)
    home = homeGames(games).loc[predictions.index]
    away = awayRows(games, home)
    rows = pd.DataFrame({"gameId": home["gameId"].values, "season": season, "date": home["date"].dt.strftime("%Y-%m-%d").values,
                         "start": (home["date"] + pd.Timedelta(hours=23)).dt.strftime("%Y-%m-%dT%H:%M:%SZ").values,
                         "home": home["team"].values, "away": away["team"].values,
                         "homeGoalie": home["goalie"].values, "homeGoalieStatus": "Actual",
                         "awayGoalie": away["goalie"].values, "awayGoalieStatus": "Actual",
                         "homeRest": home["rest"].values, "awayRest": away["rest"].values,
                         "pHome": predictions["pWin"].values, "homeRate": predictions["homeRate"].values,
                         "awayRate": predictions["awayRate"].values, "predictedAt": "", "source": "reconstructed"})
    print("Reconstructed", len(rows), "games played before logging started")
    return pd.concat([log, rows], ignore_index=True)
def fetchResults(log):
    # final scores from the NHL (shootout winners count, the decidedBy column says how the game ended)
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    pending = log[log["homeScore"].isna() & (log["date"] <= today)]
    for date in sorted(pending["date"].unique()):
        response = get(f"https://api-web.nhle.com/v1/score/{date}")
        if response.status_code != 200:
            continue
        for g in response.json().get("games", []):
            if g.get("gameState") not in ("OFF", "FINAL"):
                continue
            match = log["gameId"] == g["id"]
            log.loc[match, ["homeScore", "awayScore", "decidedBy"]] = [g["homeTeam"].get("score"), g["awayTeam"].get("score"),
                                                                      (g.get("gameOutcome") or {}).get("lastPeriodType")]
    return log
def updateSeasonLog(schedule, goalieNames, season):
    log = loadLog()
    log = backfill(log, season)
    log = logPredictions(log, schedule, goalieNames, season)
    log = fetchResults(log)
    log = log.sort_values(["date", "start", "gameId"])
    log.to_csv(LOG, index=False)
    done = log["homeScore"].notna().sum()
    print(f"Season log: {len(log)} games, {done} with results")
    return log
