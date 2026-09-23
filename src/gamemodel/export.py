import json
import pickle
import numpy as np
import pandas as pd
from src.gamemodel.download import DATA
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
def exportSite():
    with open(DATA + "game_model.pkl", "rb") as f:
        model = pickle.load(f)
    features = pd.read_csv(DATA + "features.csv", parse_dates=["date"])
    season = int(features["season"].max())
    games = features[features["season"] == season]
    teams = pd.read_csv(DATA + "team_state.csv")
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
    seasonScore = playerRankings.set_index(["name", "playerTeam"])["score"].to_dict()
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
        group = group.sort_values(["typicalLineup", "rating"], ascending=[False, False])
        playerRows[team] = [{"name": p["name"], "position": p["position"], "rating": p["rating"], "typical": bool(p["typicalLineup"]),
                             "recentGames": int(p["recentGames"]), "seasonScore": seasonScore.get((p["name"], team))}
                            for _, p in group.iterrows()]
    goalieRows = {team: group.sort_values(["recentStarts", "starts"], ascending=False)[["goalieId", "name", "starts", "recentStarts", "goalieRating"]].to_dict("records")
                  for team, group in goalies.groupby("team")}
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
        "season": season, "asOf": str(teams["lastGame"].max())[:10],
        "model": {"k": model.k, "blend": model.blend, "eloFit": model.eloFit, "players": model.players,
                  "form": model.form, "extras": list(model.extras), "timeZones": TIME_ZONES,
                  "goal": linear(model.goalScaler, model.goals, goalColumns, model.goals.coef_, model.goals.intercept_),
                  "win": linear(model.winScaler, model.win, winColumns, model.win.coef_, model.win.intercept_)},
        "teams": teamRows, "goalies": goalieRows, "players": playerRows, "trends": trends, "recent": recent,
        "report": report,
        "backtest": backtest[["date", "home", "away", "homeGoals", "awayGoals", "homeRate", "awayRate", "pWin"]].values.tolist()
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("window.NHL_DATA = " + json.dumps(clean(data), separators=(",", ":")) + ";\n")
    print("Wrote", OUTPUT)
if __name__ == "__main__":
    exportSite()
