import pandas as pd
class Player:
    def __init__(self, name, stats, team):
        self.name = name
        self.stats = stats
        self.team = team
    def calcPowerScore(self, model, scaler, featureNames):
        # get the player's data as a df
        values = pd.DataFrame([self.stats])
        # filter for only the features the model uses
        values = values[featureNames]
        # scale the data in the values df
        values = scaler.transform(values)
        # even though the model is only predicting one value(GAR), it is still an array
        # return the first value in the array
        return model.predict(values)[0]
    def __str__(self):
        return self.name