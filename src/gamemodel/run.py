import sys
from src.gamemodel.build import buildFeatures
from src.gamemodel.train import trainModel, refitModel
from src.gamemodel.export import exportSite
# rebuild everything for the game model and the website data
# python -m src.gamemodel.run          full: download/sync data, build features, backtest every model version, train, export
# python -m src.gamemodel.run --daily  daily: sync new games, build features, refit the chosen model, export (used by the GitHub Action)
if __name__ == "__main__":
    buildFeatures()
    if "--daily" in sys.argv:
        refitModel()
    else:
        trainModel()
    exportSite()
