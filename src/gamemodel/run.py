from src.gamemodel.build import buildFeatures
from src.gamemodel.train import trainModel
from src.gamemodel.export import exportSite
# rebuild everything for the game model: download data (first run only), build features, backtest and train, export the webpage data
# python -m src.gamemodel.run
if __name__ == "__main__":
    buildFeatures()
    trainModel()
    exportSite()
