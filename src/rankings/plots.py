import matplotlib.pyplot as plt
import numpy as np
def plotItems(items, model, scaler, featureNames, top = None, title = "Rankings"):
    names = []
    scores = []
    for item in items:
        names.append(item.name)
        scores.append(item.calcPowerScore(model, scaler, featureNames))
    # plot the scores using the proper top
    if top:
        names = names[0:top]
        scores = scores[0:top]
        mu = np.mean(scores)
        sd = np.std(scores)
        elite = mu + 1.5 * sd
        good = mu + 0.5 * sd
        average = mu
        bad = mu - 0.5 * sd 
        terrible = mu - 1.5 * sd
        fig, ax = plt.subplots()
        # add colored sections based on how many deviations from the mean
        ax.axhspan(elite, max(scores), color="darkgreen", alpha=0.2, label="Elite")
        ax.axhspan(good, elite, color="green", alpha=0.2, label="Good")
        ax.axhspan(average, good, color="yellow", alpha=0.2, label="Average")
        ax.axhspan(bad, average, color="orange", alpha=0.2, label="Below Average")
        ax.axhspan(terrible, bad, color="red", alpha=0.2, label="Bad")
        ax.axhspan(min(scores), terrible, color="darkred", alpha=0.2, label="Terrible")
        ax.scatter(names, scores)
        ax.set_title(title)
        ax.set_ylabel("PowerScore")
        ax.set_xlabel("Name")
        plt.tight_layout()
        plt.show()