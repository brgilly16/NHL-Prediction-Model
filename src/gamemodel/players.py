import numpy as np
import pandas as pd
from src.model.calcweights import calcWeightsPlayer
from src.gamemodel.download import DATA, loadSkaters
# player PowerScores going into every game, and each team's lineup / missing player power for every game
# a player's rating is the existing player PowerScore model (predicted GAR) applied to his recency weighted career stats
CODE_MAP = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
PLAYER_HALFLIFE = 60
PLAYER_PRIOR_GAMES = 10
SEASON_GAMES = 82
# the columns src/model/data.py playersFilter divides by games played, everything else stays a season total
PER_GAME_COLUMNS = [
    "OnIce_F_goals", "OnIce_A_goals", "OnIce_F_xGoals", "OnIce_A_xGoals", "OnIce_F_shotsOnGoal", "OnIce_A_shotsOnGoal",
    "OnIce_F_shotAttempts", "OnIce_A_shotAttempts", "OnIce_F_unblockedShotAttempts", "OnIce_A_unblockedShotAttempts",
    "OnIce_F_highDangerShots", "OnIce_A_highDangerShots", "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
    "I_F_points", "I_F_shotsOnGoal", "I_F_shotAttempts", "I_F_unblockedShotAttempts", "I_F_missedShots",
    "I_F_blockedShotAttempts", "I_F_takeaways", "I_F_giveaways", "I_F_hits", "I_F_rebounds", "I_F_reboundGoals",
    "I_F_savedShotsOnGoal"
]
# percentage columns are ratios of summed stats
PERCENT_COLUMNS = {
    "onIce_xGoalsPercentage": ("OnIce_F_xGoals", "OnIce_A_xGoals"),
    "offIce_xGoalsPercentage": ("OffIce_F_xGoals", "OffIce_A_xGoals"),
    "onIce_corsiPercentage": ("OnIce_F_shotAttempts", "OnIce_A_shotAttempts"),
    "offIce_corsiPercentage": ("OffIce_F_shotAttempts", "OffIce_A_shotAttempts"),
    "onIce_fenwickPercentage": ("OnIce_F_unblockedShotAttempts", "OnIce_A_unblockedShotAttempts"),
    "offIce_fenwickPercentage": ("OffIce_F_unblockedShotAttempts", "OffIce_A_unblockedShotAttempts")
}
class PlayerScorer:
    # the player PowerScore model written as weights so it can score many stat lines at once
    def __init__(self):
        model, scaler, featureNames = calcWeightsPlayer()
        weights = model.coef_ / scaler.scale_
        self.intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
        # features the model gives zero weight (Lasso drops many) are skipped
        used = weights != 0
        self.featureNames = [f for f, u in zip(featureNames, used) if u]
        self.weights = weights[used]
        self.columns = sorted({c for f in self.featureNames for c in PERCENT_COLUMNS.get(f, (f,))})
    def score(self, sums, games, pace=None):
        # sums: summed stats (rows x self.columns), games: games in each stat line
        # pace scales season totals to a fixed number of games, so part-season stat lines are comparable to full seasons
        index = {c: i for i, c in enumerate(self.columns)}
        games = np.asarray(games, dtype=float)
        safe = np.where(games > 0, games, 1)
        features = np.empty((len(sums), len(self.featureNames)))
        for j, f in enumerate(self.featureNames):
            if f in PERCENT_COLUMNS:
                a, b = (sums[:, index[c]] for c in PERCENT_COLUMNS[f])
                features[:, j] = np.divide(a, a + b, out=np.full(len(a), 0.5), where=(a + b) > 0)
            elif f in PER_GAME_COLUMNS:
                features[:, j] = sums[:, index[f]] / safe
            else:
                features[:, j] = sums[:, index[f]] if pace is None else sums[:, index[f]] / safe * pace
        return self.intercept + features @ self.weights
def loadSkaterGames(scorer):
    df = loadSkaters()
    df = df[["playerId", "season", "name", "gameId", "playerTeam", "gameDate", "position", "icetime"] + scorer.columns]
    df["team"] = df["playerTeam"].replace(CODE_MAP)
    df["date"] = pd.to_datetime(df["gameDate"].astype(str))
    return df.drop(columns=["playerTeam", "gameDate"]).drop_duplicates(["playerId", "gameId"]).sort_values(["playerId", "date", "gameId"]).reset_index(drop=True)
def playerRatings(df, scorer):
    # rating before each game (pre) and after it (post) from recency weighted stats, shrunk toward replacement level (0 GAR)
    stats = df[scorer.columns].to_numpy(dtype=float)
    ids = df["playerId"].to_numpy()
    decay = 0.5 ** (1 / PLAYER_HALFLIFE)
    preSums, postSums = np.zeros_like(stats), np.zeros_like(stats)
    preGames, postGames = np.zeros(len(df)), np.zeros(len(df))
    sums, games, last = np.zeros(stats.shape[1]), 0.0, None
    for i in range(len(df)):
        if ids[i] != last:
            sums, games, last = np.zeros(stats.shape[1]), 0.0, ids[i]
        preSums[i], preGames[i] = sums, games
        sums = decay * sums + stats[i]
        games = decay * games + 1
        postSums[i], postGames[i] = sums, games
    shrink = lambda g: g / (g + PLAYER_PRIOR_GAMES)
    df["ratingPre"] = np.where(preGames > 0, scorer.score(preSums, preGames, SEASON_GAMES) * shrink(preGames), 0.0)
    df["ratingPost"] = scorer.score(postSums, postGames, SEASON_GAMES) * shrink(postGames)
    return df
def teamLineups(df, teamGames):
    # lineup power = sum of the ratings of the skaters who dressed, missing power = positive ratings of regulars who did not
    lineup = df.groupby(["gameId", "team"])["ratingPre"].sum().rename("lineupPower").reset_index()
    # roster: a player belongs to a team from his first to his last game with it that season (known before each game)
    spans = df.groupby(["playerId", "team", "season"])["date"].agg(first="min", last="max").reset_index()
    games = teamGames[["gameId", "team", "season", "date"]]
    roster = games.merge(spans, on=["team", "season"])
    roster = roster[(roster["date"] >= roster["first"]) & (roster["date"] <= roster["last"])]
    played = df[["gameId", "playerId"]].assign(played=1)
    roster = roster.merge(played, on=["gameId", "playerId"], how="left").fillna({"played": 0})
    roster = roster.sort_values(["playerId", "team", "season", "date"])
    group = roster.groupby(["playerId", "team", "season"])
    roster["playedBefore"] = group["played"].cumsum() - roster["played"]
    roster["gamesBefore"] = group.cumcount()
    # regulars have played at least half of the team's games since joining (and at least 3)
    regular = (roster["playedBefore"] >= 3) & (roster["playedBefore"] >= 0.5 * roster["gamesBefore"])
    absent = roster[(roster["played"] == 0) & regular].sort_values("date")
    # an absent player's rating is his rating after his last game before this date
    history = df[["playerId", "date", "ratingPost", "name"]].sort_values("date")
    absent = pd.merge_asof(absent, history, on="date", by="playerId", allow_exact_matches=False)
    absent["value"] = absent["ratingPost"].fillna(0).clip(lower=0)
    missing = absent.groupby(["gameId", "team"]).agg(missingPower=("value", "sum")).reset_index()
    result = games.merge(lineup, on=["gameId", "team"], how="left").merge(missing, on=["gameId", "team"], how="left")
    result["missingPower"] = result["missingPower"].fillna(0)
    return result, absent
def addLineupFeatures(teamGames):
    scorer = PlayerScorer()
    skaters = playerRatings(loadSkaterGames(scorer), scorer)
    lineups, absent = teamLineups(skaters, teamGames)
    df = teamGames.merge(lineups[["gameId", "team", "lineupPower", "missingPower"]], on=["gameId", "team"], how="left")
    # lineup compared with the team's recent lineups (so it captures both injuries and stars returning)
    df = df.sort_values(["franchise", "date", "gameId"])
    df["lineupPower"] = df["lineupPower"].fillna(df.groupby("franchise")["lineupPower"].transform("mean"))
    typical = df.groupby("franchise")["lineupPower"].transform(lambda s: s.ewm(halflife=10).mean())
    df["lineupTypical"] = typical.groupby(df["franchise"]).shift(1)
    df["lineupDelta"] = (df["lineupPower"] - df["lineupTypical"]).fillna(0)
    return df, skaters
def playerState(skaters, teamGames, season, rosters=None):
    # each skater's current rating and each team's projected lineup (12 forwards and 6 defensemen)
    # players are placed on the team from the current NHL roster (so offseason trades and signings count right away);
    # without a roster a player stays with the team he last played for
    last = skaters.sort_values("date").groupby("playerId").tail(1).set_index("playerId")
    current = last[last["season"] >= season - 1][["name", "team", "position", "ratingPost"]].reset_index()
    if rosters is not None and len(rosters):
        skaterRoster = rosters[rosters["position"] != "G"]
        # players not on any fetched roster (unsigned, retired, in the minors) are dropped for the teams that were fetched
        current = current[~current["team"].isin(skaterRoster["team"].unique()) & ~current["playerId"].isin(skaterRoster["playerId"])]
        onRoster = skaterRoster.merge(last[["ratingPost"]], left_on="playerId", right_index=True, how="left")
        # players with no NHL games yet (rookies, signings from other leagues) start at replacement level
        onRoster["ratingPost"] = onRoster["ratingPost"].fillna(0.0)
        current = pd.concat([current, onRoster[["playerId", "name", "team", "position", "ratingPost"]]], ignore_index=True)
    # expected role: average ice time over each player's last 20 games (any team); games in his team's last 10 games
    history = skaters[skaters["season"] >= season - 1].sort_values("date").groupby("playerId").tail(20)
    role = history.groupby("playerId")["icetime"].mean().rename("recentIcetime")
    recentGames = teamGames.sort_values("date").groupby("team").tail(10)
    recent = skaters.merge(recentGames[["gameId", "team"]], on=["gameId", "team"])
    usage = recent.groupby(["team", "playerId"]).size().rename("recentGames")
    current = current.join(role, on="playerId").join(usage, on=["team", "playerId"]).fillna({"recentGames": 0, "recentIcetime": 0})
    current["recentGames"] = current["recentGames"].astype(int)
    # once a team has played 5 games this season, who has actually been dressing decides the lineup; before that, ice time
    started = teamGames[teamGames["season"] == season].groupby("team").size()
    current["inSeason"] = current["team"].map(started).fillna(0) >= 5
    current["group"] = np.where(current["position"] == "D", "D", "F")
    current["sortGames"] = np.where(current["inSeason"], current["recentGames"], 0)
    current = current.sort_values(["team", "group", "sortGames", "recentIcetime"], ascending=[True, True, False, False])
    current["depth"] = current.groupby(["team", "group"]).cumcount()
    current["typicalLineup"] = np.where(current["group"] == "D", current["depth"] < 6, current["depth"] < 12)
    current = current.rename(columns={"ratingPost": "rating"}).drop(columns=["sortGames", "inSeason"])
    current.to_csv(DATA + "player_state.csv", index=False)
    return current
