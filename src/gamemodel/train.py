import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
from src.gamemodel.download import DATA
# the game prediction model
# goals: each team's goals in a game are modeled as Poisson, with the rate depending on both teams' pregame features
# win probability: a logistic model on the same features, blended with the win probability implied by the goal model
# everything is validated walk-forward: each test season is predicted by a model trained only on earlier seasons
FIRST_SEASON = 2009
TEST_SEASONS = list(range(2017, 2026))
MAX_GOALS = 12
def blendPower(raw, previous, games, k):
    # early in the season a team's PowerScore is based on few games, so lean on last season's final PowerScore
    raw = raw.fillna(previous)
    return (games * raw + k * previous) / (games + k) if k > 0 else raw
FORM_PREFIX = {5: "form5_", 10: "form_", 20: "form20_"}
TIERS = [("Toss-up", 0.50), ("Lean", 0.55), ("Solid", 0.60), ("Strong", 0.65)]
def goalFeatures(df, k, eloFit=None, players=None, form=10, extras=()):
    # features for a row where "team" is scoring against "opponent"
    league = df["leagueGoals"]
    f = FORM_PREFIX[form]
    X = pd.DataFrame({
        "power": blendPower(df["powerRaw"], df["powerPrev"], df["gamesPlayed"], k),
        "oppPower": blendPower(df["opp_powerRaw"], df["opp_powerPrev"], df["opp_gamesPlayed"], k),
        "elo": df["elo"],
        "oppElo": df["opp_elo"],
        "logXGF": np.log(df[f + "xGoalsFor"].fillna(league)),
        "logXGA": np.log(df[f + "xGoalsAgainst"].fillna(league)),
        "logGF": np.log(df[f + "goalsFor"].fillna(league)),
        "logOppXGF": np.log(df["opp_" + f + "xGoalsFor"].fillna(league)),
        "logOppXGA": np.log(df["opp_" + f + "xGoalsAgainst"].fillna(league)),
        "logOppGA": np.log(df["opp_" + f + "goalsAgainst"].fillna(league)),
        "goalie": df["goalieRating"],
        "oppGoalie": df["opp_goalieRating"],
        "home": df["home"],
        "backToBack": df["backToBack"],
        "oppBackToBack": df["opp_backToBack"],
        "logLeague": np.log(league)
    }, index=df.index)
    if eloFit == "none":
        X = X.drop(columns=["elo", "oppElo"])
    elif eloFit is not None:
        # Elo and PowerScore mostly measure the same thing, so Elo only keeps the part PowerScore does not explain
        intercept, slope = eloFit
        X["elo"] = X["elo"] - (intercept + slope * X["power"])
        X["oppElo"] = X["oppElo"] - (intercept + slope * X["oppPower"])
    # player based features: regulars missing (sum of their positive player PowerScores) and tonight's lineup vs the team's usual lineup
    if players in ("missing", "both"):
        X["missing"] = df["missingPower"].fillna(0)
        X["oppMissing"] = df["opp_missingPower"].fillna(0)
    if players in ("lineup", "both"):
        X["lineupDelta"] = df["lineupDelta"].fillna(0)
        X["oppLineupDelta"] = df["opp_lineupDelta"].fillna(0)
    for side, prefix in (("", ""), ("opp", "opp_")):
        cap = (lambda name: side + name[0].upper() + name[1:]) if side else (lambda name: name)
        if "adj" in extras:
            # score and venue adjusted xG form (removes the effect of teams sitting on leads)
            X[cap("logAdjXGF")] = np.log(df[prefix + "form_adj_xGF"].fillna(league))
            X[cap("logAdjXGA")] = np.log(df[prefix + "form_adj_xGA"].fillna(league))
        if "ev" in extras:
            # 5 on 5 xG per 60 minutes
            ice = df[prefix + "form_ev_ice"].where(df[prefix + "form_ev_ice"] > 0)
            X[cap("logEvXGF60")] = np.log((df[prefix + "form_ev_xGF"] / ice * 3600).fillna(2.4).clip(lower=0.5))
            X[cap("logEvXGA60")] = np.log((df[prefix + "form_ev_xGA"] / ice * 3600).fillna(2.4).clip(lower=0.5))
        if "st" in extras:
            # special teams: power play xG created and penalty kill xG allowed per game (opportunities times efficiency)
            X[cap("ppXG")] = df[prefix + "form_pp_xGF"].fillna(0.5)
            X[cap("pkXG")] = df[prefix + "form_pk_xGA"].fillna(0.5)
        if "travel" in extras:
            X[cap("roadTrip")] = df[prefix + "roadTrip"].fillna(0).clip(upper=6)
            X[cap("tzShift")] = df[prefix + "tzShift"].fillna(0)
            X[cap("tzFromHome")] = df[prefix + "tzFromHome"].fillna(0)
    return X
def winFeatures(df, k, eloFit=None, players=None, form=10, extras=()):
    # features from the home team's point of view (home is constant so it is dropped, the intercept is home ice)
    return goalFeatures(df, k, eloFit, players, form, extras).drop(columns=["home"])
def loadFeatures():
    df = pd.read_csv(DATA + "features.csv", parse_dates=["date"])
    df = df[df["season"] >= FIRST_SEASON].dropna(subset=["opp_elo"]).reset_index(drop=True)
    return df
def homeGames(df):
    # one row per game from the home team's side, with the result
    home = df[df["home"] == 1].copy()
    home["homeWin"] = np.where(home["goalsFor"] > home["goalsAgainst"], 1.0,
                               np.where(home["goalsFor"] < home["goalsAgainst"], 0.0, 0.5))
    return home
def awayRows(df, home):
    # the away team's row for each home game, lined up with the home rows
    away = df[df["home"] == 0].set_index("gameId")
    return away.loc[home["gameId"]]
def winFromGoals(homeRate, awayRate):
    # probability the home team outscores the away team, a tie in all situations goals means a shootout (coin flip)
    goals = np.arange(MAX_GOALS + 1)
    h = poisson.pmf(goals[None, :], np.asarray(homeRate)[:, None])
    a = poisson.pmf(goals[None, :], np.asarray(awayRate)[:, None])
    joint = h[:, :, None] * a[:, None, :]
    homeAhead = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)
    tie = np.eye(MAX_GOALS + 1)
    return (joint * homeAhead).sum(axis=(1, 2)) + 0.5 * (joint * tie).sum(axis=(1, 2))
def logLoss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def winMetrics(y, p):
    decided = y != 0.5
    return {
        "logLoss": logLoss(y, p),
        "accuracy": float(np.mean((p[decided] > 0.5) == (y[decided] == 1))),
        "brier": float(np.mean((p - y) ** 2))
    }
def expandShootouts(X, y):
    # shootout games count as half a win and half a loss when training the classifier
    tie = y == 0.5
    X2 = pd.concat([X, X[tie]])
    y2 = np.concatenate([np.where(tie, 1, y), np.zeros(tie.sum())])
    weights = np.concatenate([np.where(tie, 0.5, 1.0), np.full(tie.sum(), 0.5)])
    return X2, y2.astype(int), weights
class GameModel:
    def __init__(self, k=10, goalModel="glm", alpha=1e-4, C=1.0, blend=0.5, elo="residual", players=None, form=10, extras=()):
        self.k, self.goalModel, self.alpha, self.C, self.blend, self.elo = k, goalModel, alpha, C, blend, elo
        self.players, self.form, self.extras = players, form, tuple(extras)
        self.eloFit = None
    def features(self, df):
        return goalFeatures(df, self.k, self.eloFit, self.players, self.form, self.extras)
    def homeFeatures(self, df):
        return winFeatures(df, self.k, self.eloFit, self.players, self.form, self.extras)
    def fit(self, df):
        if self.elo == "none":
            self.eloFit = "none"
        elif self.elo == "residual":
            power = blendPower(df["powerRaw"], df["powerPrev"], df["gamesPlayed"], self.k)
            slope, intercept = np.polyfit(power, df["elo"], 1)
            self.eloFit = (float(intercept), float(slope))
        X = self.features(df)
        self.goalScaler = StandardScaler().fit(X)
        if self.goalModel == "glm":
            self.goals = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        else:
            self.goals = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300,
                                                       max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=42)
        self.goals.fit(self.goalScaler.transform(X), df["goalsFor"])
        home = homeGames(df)
        W = self.homeFeatures(home)
        self.winScaler = StandardScaler().fit(W)
        Wx, y, weights = expandShootouts(pd.DataFrame(self.winScaler.transform(W), index=W.index), home["homeWin"].values)
        self.win = LogisticRegression(C=self.C, max_iter=1000)
        self.win.fit(Wx, y, sample_weight=weights)
        return self
    def predictGoals(self, df):
        return self.goals.predict(self.goalScaler.transform(self.features(df)))
    def predictGames(self, df):
        # df holds home rows and away rows, returns one prediction per home row
        home = df[df["home"] == 1]
        away = awayRows(df, home)
        homeRate = self.predictGoals(home)
        awayRate = self.predictGoals(away)
        pLogistic = self.win.predict_proba(self.winScaler.transform(self.homeFeatures(home)))[:, 1]
        pGoals = winFromGoals(homeRate, awayRate)
        pWin = self.blend * pLogistic + (1 - self.blend) * pGoals
        return pd.DataFrame({"gameId": home["gameId"].values, "homeRate": homeRate, "awayRate": awayRate,
                             "pLogistic": pLogistic, "pGoals": pGoals, "pWin": pWin}, index=home.index)
def walkForward(df, **params):
    # predictions for every test season from a model trained only on earlier seasons
    results = []
    for season in TEST_SEASONS:
        model = GameModel(**params).fit(df[df["season"] < season])
        test = df[df["season"] == season]
        predictions = model.predictGames(test)
        predictions["season"] = season
        results.append(predictions)
    return pd.concat(results)
def evaluate(df, predictions, name):
    home = homeGames(df).loc[predictions.index]
    away = awayRows(df, home)
    y = home["homeWin"].values
    goals = np.concatenate([home["goalsFor"].values, away["goalsFor"].values])
    rates = np.concatenate([predictions["homeRate"].values, predictions["awayRate"].values])
    result = {"model": name, **winMetrics(y, predictions["pWin"].values),
              "goalDeviance": float(mean_poisson_deviance(goals, rates)), "goalMAE": float(mean_absolute_error(goals, rates))}
    print(f"{name:40s} logLoss {result['logLoss']:.4f}  acc {result['accuracy']:.4f}  brier {result['brier']:.4f}  "
          f"goalDev {result['goalDeviance']:.4f}  goalMAE {result['goalMAE']:.4f}")
    return result
def baselines(df):
    # simple reference points the model has to beat
    home = homeGames(df)
    test = home[home["season"].isin(TEST_SEASONS)]
    y = test["homeWin"].values
    results = []
    homeRate = np.array([home[home["season"] < s]["homeWin"].mean() for s in test["season"]])
    results.append({"model": "Always home ice rate", **winMetrics(y, homeRate)})
    elo = 1 / (1 + 10 ** (-(test["elo"] + 35 - test["opp_elo"]) / 400))
    results.append({"model": "Elo only", **winMetrics(y, elo.values)})
    # PowerScore difference alone (plus home ice) in a logistic model, walk-forward
    p = np.empty(len(test))
    for season in TEST_SEASONS:
        train = home[home["season"] < season]
        mask = (test["season"] == season).values
        X = lambda d: (blendPower(d["powerRaw"], d["powerPrev"], d["gamesPlayed"], 10) -
                       blendPower(d["opp_powerRaw"], d["opp_powerPrev"], d["opp_gamesPlayed"], 10)).to_frame()
        Xt, yt, w = expandShootouts(X(train), train["homeWin"].values)
        p[mask] = LogisticRegression().fit(Xt, yt, sample_weight=w).predict_proba(X(test[mask]))[:, 1]
    results.append({"model": "PowerScore only", **winMetrics(y, p)})
    for r in results:
        print(f"{r['model']:40s} logLoss {r['logLoss']:.4f}  acc {r['accuracy']:.4f}  brier {r['brier']:.4f}")
    return results
def tierOf(p):
    confidence = max(p, 1 - p)
    return [name for name, cut in TIERS if confidence >= cut][-1]
def tierReport(df, predictions):
    # how often the pick is right in each confidence tier, over every backtest season and the latest one
    home = homeGames(df).loc[predictions.index]
    games = pd.DataFrame({"season": predictions["season"].values, "p": predictions["pWin"].values, "y": home["homeWin"].values})
    games = games[games["y"] != 0.5]
    games["tier"] = games["p"].map(tierOf)
    games["correct"] = (games["p"] >= 0.5) == (games["y"] == 1)
    report = []
    for name, cut in TIERS:
        row = {"tier": name, "minConfidence": cut}
        for label, subset in (("all", games), ("latest", games[games["season"] == TEST_SEASONS[-1]])):
            tier = subset[subset["tier"] == name]
            row[label] = {"games": int(len(tier)), "share": float(len(tier) / len(subset)), "accuracy": float(tier["correct"].mean())}
        report.append(row)
        print(f"{name:8s} all seasons {row['all']['accuracy']:.3f} ({row['all']['share']:.0%} of games)   latest {row['latest']['accuracy']:.3f} ({row['latest']['share']:.0%})")
    return report
def trainModel():
    df = loadFeatures()
    print("Rows:", len(df), "Seasons:", df["season"].min(), "-", df["season"].max())
    report = {"baselines": baselines(df), "candidates": []}
    # candidate models, each scored walk-forward
    # candidate models, each scored walk-forward (parameters not listed keep the GameModel defaults)
    lineup = {"players": "lineup"}
    candidates = [
        {"elo": "raw"}, {"elo": "none"}, {"elo": "residual"}, {"k": 0}, {"k": 20}, {"blend": 1.0}, {"blend": 0.0},
        {"alpha": 1e-2, "C": 0.01}, {"goalModel": "hgb"}, {"players": "missing"}, lineup, {"players": "both"},
        # newer feature groups, each tried on top of the lineup model
        lineup | {"form": 5}, lineup | {"form": 20},
        lineup | {"extras": ["adj"]}, lineup | {"extras": ["ev"]}, lineup | {"extras": ["st"]}, lineup | {"extras": ["travel"]}
    ]
    results = {}
    best, bestScore, bestPredictions = None, np.inf, None
    def score(params):
        nonlocal best, bestScore, bestPredictions
        predictions = walkForward(df, **params)
        result = evaluate(df, predictions, json.dumps(params))
        result["params"] = params
        report["candidates"].append(result)
        results[json.dumps(params)] = result["logLoss"]
        if result["logLoss"] < bestScore:
            best, bestScore, bestPredictions = params, result["logLoss"], predictions
    for params in candidates:
        score(params)
    # combine every feature group and form speed that beat the lineup model on its own
    base = results[json.dumps(lineup)]
    helped = [g for g in ["adj", "ev", "st", "travel"] if results[json.dumps(lineup | {"extras": [g]})] < base]
    forms = {f: results[json.dumps(lineup | {"form": f})] for f in (5, 20)}
    bestForm = min(forms, key=forms.get) if min(forms.values()) < base else 10
    combined = lineup | ({"form": bestForm} if bestForm != 10 else {}) | ({"extras": helped} if helped else {})
    if json.dumps(combined) not in results:
        score(combined)
    print("Best:", best)
    report["best"] = best
    report["bestBySeason"] = [evaluate(df[df["season"] == s], bestPredictions[bestPredictions["season"] == s], f"season {s}") | {"season": s}
                              for s in TEST_SEASONS]
    # out of sample predictions for the latest season, for the game log on the webpage
    latest = bestPredictions[bestPredictions["season"] == TEST_SEASONS[-1]]
    home = homeGames(df).loc[latest.index]
    away = awayRows(df, home)
    log = latest.assign(date=home["date"].dt.strftime("%Y-%m-%d").values, home=home["team"].values, away=away["team"].values,
                        homeGoals=home["goalsFor"].values, awayGoals=away["goalsFor"].values,
                        homeGoalie=home["goalie"].values, awayGoalie=away["goalie"].values)
    log.to_csv(DATA + "backtest_latest_season.csv", index=False)
    report["tiers"] = tierReport(df, bestPredictions)
    # final model uses every season
    final = GameModel(**best).fit(df)
    if final.goalModel == "glm":
        report["goalCoefficients"] = dict(zip(final.features(df.head()).columns, final.goals.coef_.round(4).tolist()))
    report["winCoefficients"] = dict(zip(final.homeFeatures(df.head()).columns, final.win.coef_[0].round(4).tolist()))
    with open(DATA + "game_model.pkl", "wb") as f:
        pickle.dump(final, f)
    with open(DATA + "model_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("Saved model to", DATA + "game_model.pkl")
    return final
def refitModel():
    # daily update: refit the already chosen model on every game (including new ones) without re-running the backtest
    with open(DATA + "model_report.json") as f:
        report = json.load(f)
    df = loadFeatures()
    final = GameModel(**report["best"]).fit(df)
    with open(DATA + "game_model.pkl", "wb") as f:
        pickle.dump(final, f)
    print("Refit model on", len(df), "rows through", df["date"].max().date())
    return final
if __name__ == "__main__":
    # import through the package so the saved model can be loaded by src.gamemodel.predict
    from src.gamemodel.train import trainModel as run
    run()
