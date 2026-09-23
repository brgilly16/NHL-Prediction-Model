import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.prediction.features import getFeatures, loadDailyPower
def createTrainingData():
    gameData = pd.read_csv("data/game_data.csv")
    dailyPower = loadDailyPower()
    trainingData = []
    for game in gameData["game"].unique():
        gameTeams = gameData[gameData["game"] == game]
        if len(gameTeams) != 2:
            continue
        teamOne = gameTeams.iloc[0]["team"]
        teamTwo = gameTeams.iloc[1]["team"]
        teamOneGoals = gameTeams.iloc[0]["goals"]
        teamTwoGoals = gameTeams.iloc[1]["goals"]
        teamOneScore, teamOneMissing = getFeatures(teamOne, game, dailyPower, gameData)
        teamTwoScore, teamTwoMissing = getFeatures(teamTwo, game, dailyPower, gameData)
        # skip games where a team has no PowerScore yet (its first game of the season)
        if pd.isna(teamOneScore) or pd.isna(teamTwoScore):
            continue
        trainingData.append({
            "teamScore": teamOneScore,
            "teamMissing": teamOneMissing,
            "goals" : teamOneGoals,
            "opponentScore": teamTwoScore,
            "opponentMissing": teamTwoMissing,
        })
        trainingData.append({
                "teamScore": teamTwoScore,
                "teamMissing": teamTwoMissing,
                "goals" : teamTwoGoals,
                "opponentScore": teamOneScore,
                "opponentMissing": teamOneMissing,
        })
    return pd.DataFrame(trainingData)
def trainModel():
    trainingData = createTrainingData()
    X = trainingData[["teamScore", "teamMissing", "opponentScore", "opponentMissing"]]
    y = trainingData["goals"]
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression()  
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(Y_test, predictions)
    print(r2)
    return model
def prediction(home, away):
    gameData = pd.read_csv("data/game_data.csv")
    game = gameData[(gameData["team"] == home) & (gameData["game"].isin(gameData[gameData["team"] == away]["game"]))]
    # use the most recent game between the two teams so the PowerScores are as current as possible
    game = game.iloc[-1]["game"]
    dailyPower = loadDailyPower()
    homeScore, homeMissing = getFeatures(home, game, dailyPower, gameData)
    awayScore, awayMissing = getFeatures(away, game, dailyPower, gameData)
    homeFeatures = pd.DataFrame([{
    "teamScore": homeScore,
    "teamMissing": homeMissing,
    "opponentScore": awayScore,
    "opponentMissing": awayMissing
    }])
    awayFeatures = pd.DataFrame([{
    "teamScore": awayScore,
    "teamMissing": awayMissing,
    "opponentScore": homeScore,
    "opponentMissing": homeMissing
    }])
    model = trainModel()
    homeGoals = model.predict(homeFeatures)[0]
    awayGoals = model.predict(awayFeatures)[0]
    return homeGoals, awayGoals