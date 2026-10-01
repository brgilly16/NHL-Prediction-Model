import os
import re
import json
import time as clock
import pickle
import numpy as np
import pandas as pd
import hashlib
from src.gamemodel.download import DATA, get, fetchStarters
from src.gamemodel.players import nameKey
from src.gamemodel.train import GameModel
from src.gamemodel.build import TIME_ZONES
# writes webapp/data.js: the trained model's weights plus every team's, goalie's and player's current state
# the webpage runs the model itself in JavaScript, so it works as a static page with no server
OUTPUT = "webapp/data.js"
def linear(scaler, model, columns, coef, intercept):
    return {"features": list(columns), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": np.ravel(coef).tolist(), "intercept": float(np.ravel(intercept)[0])}
def clean(value):
    # json safe (NaN becomes null, numpy numbers become python numbers)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    return value
def fetchSchedule(features):
    # the next two weeks of regular season games (the NHL's public schedule), with each team's days of rest before each game
    games, start = [], "now"
    for week in range(2):
        response = get(f"https://api-web.nhle.com/v1/schedule/{start}")
        if response.status_code != 200:
            print("Could not fetch the schedule")
            break
        data = response.json()
        for day in data.get("gameWeek", []):
            for g in day["games"]:
                if g["gameType"] == 2:
                    games.append({"id": g["id"], "date": day["date"], "start": g["startTimeUTC"],
                                  "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"], "state": g.get("gameState")})
        start = data.get("nextStartDate")
        if not start:
            break
    games = sorted({g["id"]: g for g in games}.values(), key=lambda g: (g["date"], g["start"]))
    # rest: days since the team's previous game (played or scheduled), capped at 4 like the training data
    previous = features.groupby("team")["date"].max().dt.strftime("%Y-%m-%d").to_dict()
    for g in games:
        for side in ("home", "away"):
            team = g[side]
            last = previous.get(team)
            days = (pd.Timestamp(g["date"]) - pd.Timestamp(last)).days if last else 4
            g[side + "Rest"] = int(min(max(days, 1), 4))
            previous[team] = g["date"]
    print("Scheduled games:", len(games))
    return games
def attachStarters(games, goalieRows, goalieState):
    # each scheduled game gets ESPN's starting goalies (Confirmed / Expected); a starter missing from his team's goalie list
    # (just traded or called up) is added to it, with his rating if he has NHL games
    starters = fetchStarters(sorted({g["date"] for g in games}))
    everyone = {nameKey(r["name"]): r for r in goalieState.to_dict("records")}
    for g in games:
        found = starters.get((g["date"], g["home"], g["away"]), {})
        for side in ("home", "away"):
            name, status = found.get(side, (None, None))
            g[side + "Goalie"], g[side + "GoalieStatus"] = None, "Projected"
            if not name:
                continue
            team = goalieRows.setdefault(g[side], [])
            match = next((x for x in team if nameKey(x["name"]) == nameKey(name)), None)
            if match is None:
                known = everyone.get(nameKey(name))
                match = {"goalieId": known["goalieId"] if known else -(int(hashlib.md5(nameKey(name).encode()).hexdigest()[:8], 16) + 1),
                         "name": name, "starts": 0, "recentStarts": 0,
                         "goalieRating": known["goalieRating"] if known else 0.0, "injury": None, "injuryOut": False}
                team.append(match)
            g[side + "Goalie"], g[side + "GoalieStatus"] = match["goalieId"], status or "Expected"
    return games
def exportSite():
    with open(DATA + "game_model.pkl", "rb") as f:
        model = pickle.load(f)
    features = pd.read_csv(DATA + "features.csv", parse_dates=["date"])
    # dataSeason: latest season with games played; season: the season upcoming games belong to (later than dataSeason in the preseason)
    dataSeason = int(features["season"].max())
    games = features[features["season"] == dataSeason]
    teams = pd.read_csv(DATA + "team_state.csv")
    season = int(teams["season"].max())
    goalies = pd.read_csv(DATA + "goalie_state.csv")
    players = pd.read_csv(DATA + "player_state.csv")
    rankings = pd.read_csv("data/rankings.csv")
    with open(DATA + "model_report.json") as f:
        report = json.load(f)
    backtest = pd.read_csv(DATA + "backtest_latest_season.csv")
    # the existing season rankings (your PowerScore pipeline output)
    teamRankings = rankings[(rankings["category"] == "teams") & (rankings["season"] == rankings["season"].max())]
    playerRankings = rankings[(rankings["category"] == "players") & (rankings["time"] == "regular")]
    playerRankings = playerRankings[playerRankings["season"] == playerRankings["season"].max()]
    # rankings.csv drops accented letters ("Stützle" is saved as "Sttzle"), so names are compared with every non-ASCII letter removed
    ascii = lambda name: re.sub(r"[^a-z]", "", "".join(c for c in str(name) if ord(c) < 128).lower())
    ranked = playerRankings.assign(key=playerRankings["name"].map(ascii))
    seasonScore = ranked.set_index(["key", "playerTeam"])["score"].to_dict()
    # players who changed teams are matched on name alone
    seasonScoreByName = ranked.drop_duplicates("key", keep=False).set_index("key")["score"].to_dict()
    teamRows = []
    for _, t in teams.iterrows():
        record = games[games["team"] == t["team"]]
        wins = int((record["goalsFor"] > record["goalsAgainst"]).sum())
        losses = int((record["goalsFor"] < record["goalsAgainst"]).sum())
        row = t.to_dict()
        row["lastGame"] = str(row["lastGame"])[:10]
        row["record"] = [wins, losses, len(record) - wins - losses]
        for time in ["regular", "playoffs"]:
            match = teamRankings[(teamRankings["time"] == time) & (teamRankings["name"] == t["team"])]["score"]
            row[time + "Power"] = float(match.iloc[0]) if len(match) else None
        teamRows.append(row)
    playerRows = {}
    for team, group in players.groupby("team"):
        # depth order within forwards and defense (the webpage builds the projected lineup from it)
        group = group.sort_values(["group", "depth"])
        # two players with the same name on one team (e.g. Vancouver's two Elias Petterssons) get their position added,
        # because the webpage tells players apart by name
        shared = group["name"].duplicated(keep=False)
        group = group.assign(name=np.where(shared, group["name"] + " (" + np.where(group["group"] == "D", "D", "F") + ")", group["name"]))
        playerRows[team] = [{"name": p["name"], "position": p["position"], "rating": p["rating"], "typical": bool(p["typicalLineup"]),
                             "group": p["group"], "recentGames": int(p["recentGames"]),
                             "injury": p["injury"] if isinstance(p["injury"], str) else None, "injuryOut": bool(p["injuryOut"]),
                             "injuryType": p["injuryType"] if isinstance(p["injuryType"], str) else None,
                             "returnDate": p["returnDate"] if isinstance(p["returnDate"], str) else None,
                             "seasonScore": seasonScore.get((ascii(re.sub(r" \([DF]\)$", "", p["name"])), team),
                                                            seasonScoreByName.get(ascii(re.sub(r" \([DF]\)$", "", p["name"]))))}
                            for _, p in group.iterrows()]
    # goalie_state.csv is already ordered with the likely starter first
    goalieRows = {team: group[["goalieId", "name", "starts", "recentStarts", "goalieRating", "injury", "injuryOut"]].to_dict("records")
                  for team, group in goalies.groupby("team", sort=False)}
    schedule = attachStarters(fetchSchedule(features), goalieRows, goalies)
    trends, recent = {}, {}
    for team, group in games.sort_values("date").groupby("team"):
        trends[team] = [[d.strftime("%Y-%m-%d"), p, e] for d, p, e in zip(group["date"], group["powerRaw"], group["elo"])]
        last = group.tail(10).iloc[::-1]
        recent[team] = [[d.strftime("%Y-%m-%d"), o, int(h), int(gf), int(ga), xf, xa, g] for d, o, h, gf, ga, xf, xa, g in
                        zip(last["date"], last["opponent"], last["home"], last["goalsFor"], last["goalsAgainst"],
                            last["xGoalsFor"], last["xGoalsAgainst"], last["goalie"])]
    goalColumns = model.features(features.head(2)).columns
    winColumns = model.homeFeatures(features.head(2)).columns
    data = {
        "season": season, "dataSeason": dataSeason, "preseason": season > dataSeason,
        "backtestSeason": int(report["bestBySeason"][-1]["season"]),
        "rankingsSeason": int(playerRankings["season"].max()) if len(playerRankings) else None,
        "asOf": str(teams["lastGame"].max())[:10],
        "model": {"k": model.k, "blend": model.blend, "eloFit": model.eloFit, "players": model.players,
                  "form": model.form, "extras": list(model.extras), "timeZones": TIME_ZONES,
                  "goal": linear(model.goalScaler, model.goals, goalColumns, model.goals.coef_, model.goals.intercept_),
                  "win": linear(model.winScaler, model.win, winColumns, model.win.coef_, model.win.intercept_)},
        "teams": teamRows, "goalies": goalieRows, "players": playerRows, "trends": trends, "recent": recent,
        "report": report,
        "backtest": backtest[["date", "home", "away", "homeGoals", "awayGoals", "homeRate", "awayRate", "pWin"]].values.tolist(),
        "schedule": schedule,
        "injuryReport": json.load(open(DATA + "injury_status.json")) if os.path.exists(DATA + "injury_status.json") else {"ok": False}
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("window.NHL_DATA = " + json.dumps(clean(data), separators=(",", ":")) + ";\n")
    # stamp a version on the script links so browsers load the new files right after an update instead of a cached copy
    page = open("webapp/index.html", encoding="utf-8").read()
    page = re.sub(r'src="(data|app)\.js(\?v=\d+)?"', lambda m: f'src="{m.group(1)}.js?v={int(clock.time())}"', page)
    open("webapp/index.html", "w", encoding="utf-8").write(page)
    print("Wrote", OUTPUT)
if __name__ == "__main__":
    exportSite()
