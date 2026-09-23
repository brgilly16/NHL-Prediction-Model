import numpy as np
import pandas as pd
from src.model.calcweights import calcWeightsTeam
from src.gamemodel.download import downloadAll, DATA
from src.gamemodel.players import addLineupFeatures, playerState
# builds one row per team per regular season game where every feature only uses information from BEFORE that game
# also saves each team's and goalie's current state (after their latest game) so upcoming games can be predicted
CODE_MAP = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
# relocated franchises keep their history (ratings, form, previous season PowerScore)
FRANCHISE_MAP = {"ATL": "WPG", "ARI": "UTA"}
POWER_STATS = ["xGoalsFor", "xGoalsAgainst", "giveawaysFor", "giveawaysAgainst",
               "highDangerShotsFor", "highDangerShotsAgainst", "goalsFor", "goalsAgainst"]
FORM_HALFLIFE = 10
# recent form is kept at several decay speeds so training can pick the best one
FORM_HALFLIVES = [5, 10, 20]
BASIC_FORM = ["xGoalsFor", "xGoalsAgainst", "goalsFor", "goalsAgainst"]
# score and venue adjusted xG (all situations), 5 on 5 xG and ice time, power play and penalty kill xG and ice time
EXTRA_FORM = ["adj_xGF", "adj_xGA", "ev_xGF", "ev_xGA", "ev_ice", "pp_xGF", "pp_ice", "pk_xGA", "pk_ice"]
# arena time zones (UTC offset, standard time) for travel features
TIME_ZONES = {"ANA": -8, "LAK": -8, "SJS": -8, "SEA": -8, "VGK": -8, "VAN": -8, "ARI": -7, "UTA": -7, "COL": -7, "CGY": -7,
              "EDM": -7, "CHI": -6, "DAL": -6, "MIN": -6, "NSH": -6, "STL": -6, "WPG": -6, "ATL": -5, "BOS": -5, "BUF": -5,
              "CAR": -5, "CBJ": -5, "DET": -5, "FLA": -5, "MTL": -5, "NJD": -5, "NYI": -5, "NYR": -5, "OTT": -5, "PHI": -5,
              "PIT": -5, "TBL": -5, "TOR": -5, "WSH": -5}
GOALIE_HALFLIFE = 60
GOALIE_PRIOR_GAMES = 15
ELO_K = 8
ELO_HOME = 35
ELO_REGRESS = 0.3
def loadTeamGames():
    columns = ["team", "season", "gameId", "opposingTeam", "home_or_away", "gameDate", "situation", "playoffGame", "iceTime"] + POWER_STATS
    df = pd.read_csv(DATA + "all_teams_raw.csv", usecols=columns)
    df = df[(df["situation"] == "all") & (df["playoffGame"] == 0)].drop(columns=["situation", "playoffGame"])
    df["team"] = df["team"].replace(CODE_MAP)
    df["opponent"] = df["opposingTeam"].replace(CODE_MAP)
    df["franchise"] = df["team"].replace(FRANCHISE_MAP)
    df["home"] = (df["home_or_away"] == "HOME").astype(int)
    df["date"] = pd.to_datetime(df["gameDate"].astype(str))
    df = df.drop(columns=["opposingTeam", "home_or_away", "gameDate"])
    # some 2011-2013 Phoenix home games list Phoenix as its own opponent and both teams as away, repair those rows
    broken = df["team"] == df["opponent"]
    other = df[~broken & df["gameId"].isin(df.loc[broken, "gameId"])].set_index("gameId")["team"]
    df.loc[broken, "opponent"] = df.loc[broken, "gameId"].map(other)
    df.loc[broken, "home"] = 1
    df = df.merge(loadSituations(), on=["gameId", "team"], how="left")
    df[EXTRA_FORM] = df[EXTRA_FORM].fillna(0)
    return df.sort_values(["franchise", "date", "gameId"]).reset_index(drop=True)
def loadSituations():
    # per game 5 on 5, power play and penalty kill numbers plus score and venue adjusted xG
    columns = ["team", "gameId", "situation", "playoffGame", "xGoalsFor", "xGoalsAgainst", "iceTime",
               "scoreVenueAdjustedxGoalsFor", "scoreVenueAdjustedxGoalsAgainst"]
    raw = pd.read_csv(DATA + "all_teams_raw.csv", usecols=columns)
    raw = raw[raw["playoffGame"] == 0]
    raw["team"] = raw["team"].replace(CODE_MAP)
    raw = raw.drop_duplicates(["gameId", "team", "situation"]).set_index(["gameId", "team", "situation"])
    def pick(situation, column):
        return raw.xs(situation, level="situation")[column]
    return pd.DataFrame({
        "adj_xGF": pick("all", "scoreVenueAdjustedxGoalsFor"), "adj_xGA": pick("all", "scoreVenueAdjustedxGoalsAgainst"),
        "ev_xGF": pick("5on5", "xGoalsFor"), "ev_xGA": pick("5on5", "xGoalsAgainst"), "ev_ice": pick("5on5", "iceTime"),
        "pp_xGF": pick("5on4", "xGoalsFor"), "pp_ice": pick("5on4", "iceTime"),
        "pk_xGA": pick("4on5", "xGoalsAgainst"), "pk_ice": pick("4on5", "iceTime")
    }).reset_index()
def powerFeatures(totals, games):
    # the exact per game features the existing team PowerScore model uses
    return pd.DataFrame({
        "XGD": (totals["xGoalsFor"] - totals["xGoalsAgainst"]) / games,
        "xGoalsPercentage": totals["xGoalsFor"] / (totals["xGoalsFor"] + totals["xGoalsAgainst"]),
        "GiveawayDifferential": (totals["giveawaysAgainst"] - totals["giveawaysFor"]) / games,
        "HDSD": (totals["highDangerShotsFor"] - totals["highDangerShotsAgainst"]) / games,
        "AGD": (totals["goalsFor"] - totals["goalsAgainst"]) / games
    })
def scorePower(features, model, scaler, featureNames):
    scores = pd.Series(np.nan, index=features.index)
    valid = features.notna().all(axis=1)
    scores[valid] = model.predict(scaler.transform(features.loc[valid, featureNames]))
    return scores
def addPowerScores(df):
    # PowerScore going into each game from the team's season-to-date stats, the same idea as src/model/dailypower.py but for every season
    model, scaler, featureNames = calcWeightsTeam()
    group = df.groupby(["franchise", "season"])
    totals = group[POWER_STATS].transform(lambda s: s.cumsum().shift(1))
    df["gamesPlayed"] = group.cumcount()
    df["powerRaw"] = scorePower(powerFeatures(totals, df["gamesPlayed"]), model, scaler, featureNames)
    # each team's final PowerScore for the season, used as the starting point for the next season
    seasonTotals = df.groupby(["franchise", "season"])[POWER_STATS].sum()
    seasonGames = df.groupby(["franchise", "season"]).size()
    final = scorePower(powerFeatures(seasonTotals, seasonGames), model, scaler, featureNames).rename("powerFinal").reset_index()
    previous = final.assign(season=final["season"] + 1).rename(columns={"powerFinal": "powerPrev"})
    df = df.merge(previous, on=["franchise", "season"], how="left")
    # expansion teams have no previous season, so they start at the league average
    df["powerPrev"] = df["powerPrev"].fillna(final["powerFinal"].mean())
    return df, final
def addForm(df):
    # exponentially weighted recent form carried across seasons, shifted so the current game is excluded
    df = df.sort_values(["franchise", "date", "gameId"])
    for name, halflife, stats in formColumns():
        ewm = df.groupby("franchise")[stats[1]].transform(lambda s: s.ewm(halflife=halflife).mean())
        df[name] = ewm.groupby(df["franchise"]).shift(1)
    # travel: games into the current road trip, and time zones crossed since the previous game and from home
    arena = np.where(df["home"] == 1, df["team"], df["opponent"])
    df["arenaZone"] = pd.Series(arena, index=df.index).map(TIME_ZONES)
    homeZone = df["team"].map(TIME_ZONES)
    previousZone = df.groupby(["franchise", "season"])["arenaZone"].shift(1).fillna(homeZone)
    df["tzShift"] = (df["arenaZone"] - previousZone).abs().fillna(0)
    df["tzFromHome"] = (df["arenaZone"] - homeZone).abs().fillna(0)
    trip = (df["home"] == 1).groupby([df["franchise"], df["season"]]).cumsum()
    df["roadTrip"] = (df["home"] == 0).astype(int).groupby([df["franchise"], df["season"], trip]).cumsum()
    df = df.drop(columns=["arenaZone"])
    # rest days since the team's previous game (capped, first game of a season counts as well rested)
    previous = df.groupby(["franchise", "season"])["date"].shift(1)
    df["rest"] = (df["date"] - previous).dt.days.fillna(4).clip(upper=4)
    df["backToBack"] = (df["rest"] == 1).astype(int)
    return df
def currentForm(df):
    # form after each team's latest game, for predicting upcoming games
    df = df.sort_values(["franchise", "date", "gameId"])
    state = {}
    for name, halflife, (stat, column) in formColumns():
        state[name] = df.groupby("franchise")[column].apply(lambda s: s.ewm(halflife=halflife).mean().iloc[-1])
    return pd.DataFrame(state)
def formColumns():
    # (feature name, half-life, (stat, source column)); form_ is the 10 game half-life, form5_ and form20_ the others
    columns = []
    for halflife in FORM_HALFLIVES:
        prefix = "form_" if halflife == FORM_HALFLIFE else f"form{halflife}_"
        columns += [(prefix + stat, halflife, (stat, stat)) for stat in BASIC_FORM]
    columns += [("form_" + stat, FORM_HALFLIFE, (stat, stat)) for stat in EXTRA_FORM]
    return columns
def formNames():
    return [name for name, _, _ in formColumns()]
def addLeagueScoring(df):
    # league scoring environment going into each date (goals per team game)
    daily = df.groupby("date")["goalsFor"].agg(["sum", "count"]).sort_index()
    ewmGoals = daily["sum"].ewm(halflife=60).mean()
    ewmGames = daily["count"].ewm(halflife=60).mean()
    leagueRate = (ewmGoals / ewmGames).shift(1)
    df["leagueGoals"] = df["date"].map(leagueRate).fillna(daily["sum"].sum() / daily["count"].sum())
    return df, float((ewmGoals / ewmGames).iloc[-1])
def addElo(df):
    # classic margin-of-victory Elo, regressed toward the mean between seasons
    home = df[df["home"] == 1].sort_values(["date", "gameId"])
    elo = {}
    lastSeason = {}
    preHome, preAway = [], []
    for row in home.itertuples():
        for team in (row.franchise, FRANCHISE_MAP.get(row.opponent, row.opponent)):
            if team not in elo:
                elo[team] = 1500.0
            elif lastSeason[team] != row.season:
                elo[team] = 1500 + (elo[team] - 1500) * (1 - ELO_REGRESS)
            lastSeason[team] = row.season
        h, a = row.franchise, FRANCHISE_MAP.get(row.opponent, row.opponent)
        preHome.append(elo[h])
        preAway.append(elo[a])
        expected = 1 / (1 + 10 ** (-(elo[h] + ELO_HOME - elo[a]) / 400))
        margin = row.goalsFor - row.goalsAgainst
        # shootouts are recorded as ties, count them as half a win
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winnerDiff = (elo[h] + ELO_HOME - elo[a]) * (1 if margin >= 0 else -1)
        multiplier = np.log(abs(margin) + 1) * 2.2 / (winnerDiff * 0.001 + 2.2) if margin != 0 else 1.0
        change = ELO_K * multiplier * (result - expected)
        elo[h] += change
        elo[a] -= change
    homeElo = pd.DataFrame({"gameId": home["gameId"].values, "eloHome": preHome, "eloAway": preAway})
    df = df.merge(homeElo, on="gameId", how="left")
    df["elo"] = np.where(df["home"] == 1, df["eloHome"], df["eloAway"])
    return df.drop(columns=["eloHome", "eloAway"]), pd.Series(elo, name="elo")
def goalieRatings():
    # rating = shrunk, recency weighted goals saved above expected per 60 minutes, going into each game
    goalies = pd.read_csv(DATA + "goalie_games.csv")
    goalies["playerTeam"] = goalies["playerTeam"].replace(CODE_MAP)
    goalies = goalies.drop_duplicates(["playerId", "gameId"]).sort_values(["playerId", "gameDate", "gameId"])
    decay = 0.5 ** (1 / GOALIE_HALFLIFE)
    pre = np.empty(len(goalies))
    current = {}
    ids = goalies["playerId"].values
    saved = (goalies["xGoals"] - goalies["goals"]).values
    hours = (goalies["icetime"] / 3600).values
    savedSum, hourSum, lastId = 0.0, 0.0, None
    for i in range(len(goalies)):
        if ids[i] != lastId:
            savedSum, hourSum, lastId = 0.0, 0.0, ids[i]
        pre[i] = savedSum / (hourSum + GOALIE_PRIOR_GAMES)
        savedSum = decay * savedSum + saved[i]
        hourSum = decay * hourSum + hours[i]
        current[ids[i]] = (savedSum / (hourSum + GOALIE_PRIOR_GAMES), hourSum)
    goalies["goalieRating"] = pre
    return goalies, current
def addGoalies(df, goalies):
    # the starter is the goalie who played the most minutes for the team that game
    starters = goalies.sort_values("icetime", ascending=False).drop_duplicates(["gameId", "playerTeam"])
    starters = starters[["gameId", "playerTeam", "playerId", "name", "goalieRating"]].rename(
        columns={"playerTeam": "team", "playerId": "goalieId", "name": "goalie"})
    df = df.merge(starters, on=["gameId", "team"], how="left")
    df["goalieRating"] = df["goalieRating"].fillna(0.0)
    return df
def addOpponent(df):
    # attach the opponent's pregame features to each row
    columns = ["powerRaw", "powerPrev", "gamesPlayed", "elo", "rest", "backToBack", "goalieRating", "goalie", "lineupPower",
               "lineupTypical", "lineupDelta", "missingPower", "roadTrip", "tzShift", "tzFromHome"] + formNames()
    opponent = df[["gameId", "team"] + columns].rename(columns={"team": "opponent"})
    opponent = opponent.rename(columns={c: "opp_" + c for c in columns})
    return df.merge(opponent, on=["gameId", "opponent"], how="left")
def buildFeatures():
    downloadAll()
    df = loadTeamGames()
    df, finalPower = addPowerScores(df)
    df = addForm(df)
    df, leagueNow = addLeagueScoring(df)
    df, elo = addElo(df)
    goalies, goalieNow = goalieRatings()
    df = addGoalies(df, goalies)
    df, skaters = addLineupFeatures(df)
    df = addOpponent(df)
    df = df.sort_values(["date", "gameId", "home"]).reset_index(drop=True)
    df.to_csv(DATA + "features.csv", index=False)
    saveCurrentState(df, finalPower, elo, goalies, goalieNow, leagueNow)
    playerState(skaters, df)
    print("Finished building features. Rows:", len(df))
    return df
def saveCurrentState(df, finalPower, elo, goalies, goalieNow, leagueNow):
    # each team's state after its latest game: PowerScore, Elo, form, and its goalies
    latestSeason = df["season"].max()
    teams = df[df["season"] == latestSeason].groupby("franchise").agg(
        team=("team", "last"), lastGame=("date", "max"), gamesPlayed=("gameId", "count"))
    power = finalPower[finalPower["season"] == latestSeason].set_index("franchise")["powerFinal"]
    previous = finalPower[finalPower["season"] == latestSeason - 1].set_index("franchise")["powerFinal"]
    teams["powerRaw"] = power
    teams["powerPrev"] = previous.reindex(teams.index).fillna(finalPower["powerFinal"].mean())
    teams["elo"] = elo.reindex(teams.index)
    teams = teams.join(currentForm(df))
    teams["leagueGoals"] = leagueNow
    # typical lineup power after the latest game
    ordered = df.sort_values(["franchise", "date", "gameId"])
    teams["lineupTypical"] = ordered.groupby("franchise")["lineupPower"].apply(lambda s: s.ewm(halflife=10).mean().iloc[-1])
    teams.reset_index().to_csv(DATA + "team_state.csv", index=False)
    # goalies who played for each team this season, with their current rating and number of starts
    season = goalies[goalies["season"] == latestSeason]
    starts = df[df["season"] == latestSeason].groupby(["team", "goalieId"]).size().rename("starts")
    recent = df[df["season"] == latestSeason].sort_values("date").groupby("team").tail(10)
    recentStarts = recent.groupby(["team", "goalieId"]).size().rename("recentStarts")
    roster = season.groupby(["playerTeam", "playerId"]).agg(name=("name", "last"), lastGame=("gameDate", "max")).reset_index()
    roster = roster.rename(columns={"playerTeam": "team", "playerId": "goalieId"})
    roster["goalieId"] = roster["goalieId"].astype(float)
    roster = roster.join(starts, on=["team", "goalieId"]).join(recentStarts, on=["team", "goalieId"]).fillna({"starts": 0, "recentStarts": 0})
    roster["goalieRating"] = roster["goalieId"].map(lambda g: goalieNow[g][0])
    roster.to_csv(DATA + "goalie_state.csv", index=False)
if __name__ == "__main__":
    buildFeatures()
