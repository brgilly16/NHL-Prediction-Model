import pandas as pd
class Team:
    def __init__(self, name, stats):
        self.name = name
        self.stats = stats
    def calcPowerScore(self, model, scaler, featureNames):
        # get the team's stats in a df
        values = pd.DataFrame([self.stats])
        # filter for only the features the model uses
        values = values[featureNames]
        # scale the values data
        values = scaler.transform(values)
        # return the prediction
        return model.predict(values)[0]
    def __str__(self):
        return self.name