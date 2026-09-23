import pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson
from src.gamemodel.download import DATA
from src.gamemodel.train import GameModel, winFromGoals
from src.gamemodel.build import TIME_ZONES
# predicts an upcoming game from each team's current state (after its latest game)
# webapp/app.js mirrors this logic in JavaScript for the webpage
FACTOR_GROUPS = {
    "PowerScore": ["power", "oppPower"],
    "Elo (beyond PowerScore)": ["elo", "oppElo"],
    "Recent form": ["logXGF", "logXGA", "logGF", "logOppXGF", "logOppXGA", "logOppGA"],
    "Goaltending": ["goalie", "oppGoalie"],
    "Rest": ["backToBack", "oppBackToBack"],
    "Missing players": ["missing", "oppMissing", "lineupDelta", "oppLineupDelta"],
    "Shot quality (5v5 / adjusted)": ["logAdjXGF", "logAdjXGA", "oppLogAdjXGF", "oppLogAdjXGA", "logEvXGF60", "logEvXGA60", "oppLogEvXGF60", "oppLogEvXGA60"],
    "Special teams": ["ppXG", "pkXG", "oppPpXG", "oppPkXG"],
    "Travel": ["roadTrip", "tzShift", "tzFromHome", "oppRoadTrip", "oppTzShift", "oppTzFromHome"],
    "Scoring environment": ["logLeague"]
}
# factors measured against zero (rested, nobody out, no travel) rather than the league average
NEUTRAL = ["backToBack", "oppBackToBack", "missing", "oppMissing", "lineupDelta", "oppLineupDelta",
           "roadTrip", "tzShift", "tzFromHome", "oppRoadTrip", "oppTzShift", "oppTzFromHome"]
class Predictor:
    def __init__(self):
        with open(DATA + "game_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.teams = pd.read_csv(DATA + "team_state.csv").set_index("team")
        self.goalies = pd.read_csv(DATA + "goalie_state.csv")
        self.players = pd.read_csv(DATA + "player_state.csv")
    def teamGoalies(self, team):
        # goalie_state.csv is already ordered with the likely starter first
        goalies = self.goalies[self.goalies["team"] == team]
        return goalies[["goalieId", "name", "starts", "recentStarts", "goalieRating"]].to_dict("records")
    def teamPlayers(self, team):
        players = self.players[self.players["team"] == team].sort_values(["typicalLineup", "group", "depth"], ascending=[False, True, True])
        return players[["name", "position", "rating", "typicalLineup", "recentGames"]].to_dict("records")
    def playersOut(self, team, names):
        # projected lineup: the top 12 forwards and 6 defensemen on the current roster who are not out (next man up fills in)
        # lineup strength compares it with the team's usual lineup, so offseason additions and losses count too
        players = self.players[self.players["team"] == team].sort_values(["group", "depth"])
        out = players["name"].isin(names or [])
        available = players[~out]
        dressed = pd.concat([available[available["group"] == "F"].head(12), available[available["group"] == "D"].head(6)])
        lineupDelta = float(dressed["rating"].sum() - self.teams.loc[team, "lineupTypical"])
        missing = float(players[out & players["typicalLineup"]]["rating"].clip(lower=0).sum())
        return missing, lineupDelta
    def teamRow(self, team, opponent, home, goalieId, rest, out):
        state = self.teams.loc[team]
        goalies = self.teamGoalies(team)
        goalie = next((g for g in goalies if g["goalieId"] == goalieId), goalies[0] if goalies else None)
        missing, lineupDelta = self.playersOut(team, out)
        # travel: the away team is assumed to come straight from home, starting a road trip
        zones = abs(TIME_ZONES[opponent if not home else team] - TIME_ZONES[team])
        form = {c: state[c] for c in state.index if c.startswith("form")}
        return form | {
            "team": team, "opponent": opponent, "home": home,
            "powerRaw": state["powerRaw"], "powerPrev": state["powerPrev"], "gamesPlayed": state["gamesPlayed"],
            "elo": state["elo"], "roadTrip": 0 if home else 1, "tzShift": zones, "tzFromHome": zones,
            "rest": rest, "backToBack": int(rest == 1), "goalieRating": goalie["goalieRating"] if goalie else 0.0,
            "goalie": goalie["name"] if goalie else None, "leagueGoals": state["leagueGoals"],
            "missingPower": missing, "lineupDelta": lineupDelta
        }
    def buildRows(self, home, away, homeGoalie, awayGoalie, homeRest, awayRest, homeOut, awayOut):
        rows = [self.teamRow(home, away, 1, homeGoalie, homeRest, homeOut), self.teamRow(away, home, 0, awayGoalie, awayRest, awayOut)]
        own = [c for c in rows[0] if c not in ("team", "opponent", "home")]
        for row, other in ((rows[0], rows[1]), (rows[1], rows[0])):
            for column in own:
                row["opp_" + column] = other[column]
        return pd.DataFrame(rows)
    def predict(self, home, away, homeGoalie=None, awayGoalie=None, homeRest=2, awayRest=2, homeMissing=None, awayMissing=None):
        model = self.model
        rows = self.buildRows(home, away, homeGoalie, awayGoalie, homeRest, awayRest, homeMissing, awayMissing)
        rates = model.predictGoals(rows)
        W = model.homeFeatures(rows.iloc[[0]])
        pLogistic = model.win.predict_proba(model.winScaler.transform(W))[:, 1]
        pGoals = winFromGoals(rates[[0]], rates[[1]])
        pWin = float((model.blend * pLogistic + (1 - model.blend) * pGoals)[0])
        # score grid from the goal model
        goals = np.arange(9)
        grid = np.outer(poisson.pmf(goals, rates[0]), poisson.pmf(goals, rates[1]))
        top = sorted(((grid[h, a], h, a) for h in goals for a in goals), reverse=True)[:5]
        # how much each group of features moves the home team's win odds (log odds), compared with an average matchup
        # where both teams are rested and nobody is missing
        baseline = pd.DataFrame([model.winScaler.mean_], columns=W.columns)
        for column in NEUTRAL:
            if column in baseline:
                baseline[column] = 0
        z = model.winScaler.transform(W)[0] - model.winScaler.transform(baseline)[0]
        contributions = dict(zip(W.columns, model.win.coef_[0] * z))
        factors = [{"factor": name, "value": float(sum(contributions.get(c, 0) for c in columns))}
                   for name, columns in FACTOR_GROUPS.items() if any(c in contributions for c in columns)]
        factors.append({"factor": "Home ice", "value": float(model.win.intercept_[0] + model.win.coef_[0] @ model.winScaler.transform(baseline)[0])})
        return {
            "home": home, "away": away,
            "homeWin": pWin, "awayWin": 1 - pWin,
            "homeGoals": float(rates[0]), "awayGoals": float(rates[1]),
            "shootout": float(np.trace(grid)),
            "topScores": [{"home": int(h), "away": int(a), "probability": float(p)} for p, h, a in top],
            "grid": grid.round(4).tolist(),
            "homeGoalie": rows.loc[0, "goalie"], "awayGoalie": rows.loc[1, "goalie"],
            "factors": factors
        }
