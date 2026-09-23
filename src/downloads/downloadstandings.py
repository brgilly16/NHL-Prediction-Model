import pandas as pd
import time
def downloadStandings():
    # for loop to get all the standings
    standings = []
    for i in range(2008, 2025):
        tables = pd.read_html(f"https://www.hockey-reference.com/leagues/NHL_{i + 1}_standings.html")
        tableOne = tables[0]
        tableTwo = tables[1] #hockey reference includes multiple different tables. we want indecies zero and one.
        tableOne = pd.concat([tableOne,tableTwo], ignore_index=True)
        tableOne = tableOne.rename(columns={"Unnamed: 0": "Team"}) # rename team column to more understandable name
        tableOne = tableOne[["Team", "PTS%"]]
        tableOne = tableOne[tableOne["Team"].str.contains("Division") == False] #remove headers
        tableOne["Team"] = tableOne["Team"].str.replace("*", "")
        teamMap = {
        "Anaheim Ducks": "ANA",
        "Boston Bruins": "BOS",
        "Buffalo Sabres": "BUF",
        "Calgary Flames": "CGY",
        "Carolina Hurricanes": "CAR",
        "Chicago Blackhawks": "CHI",
        "Colorado Avalanche": "COL",
        "Columbus Blue Jackets": "CBJ",
        "Dallas Stars": "DAL",
        "Detroit Red Wings": "DET",
        "Edmonton Oilers": "EDM",
        "Florida Panthers": "FLA",
        "Los Angeles Kings": "LAK",
        "Minnesota Wild": "MIN",
        "Montreal Canadiens": "MTL",
        "Nashville Predators": "NSH",
        "New Jersey Devils": "NJD",
        "New York Islanders": "NYI",
        "New York Rangers": "NYR",
        "Ottawa Senators": "OTT",
        "Philadelphia Flyers": "PHI",
        "Pittsburgh Penguins": "PIT",
        "San Jose Sharks": "SJS",
        "Seattle Kraken": "SEA",
        "St. Louis Blues": "STL",
        "Tampa Bay Lightning": "TBL",
        "Toronto Maple Leafs": "TOR",
        "Utah Mammoth": "UTA",
        "Utah Hockey Club": "UTA", 
        "Vancouver Canucks": "VAN",
        "Vegas Golden Knights": "VGK",
        "Washington Capitals": "WSH",
        "Winnipeg Jets": "WPG",
        "Arizona Coyotes": "ARI",
        "Phoenix Coyotes": "ARI",
        "Atlanta Thrashers": "ATL",
        }
        tableOne["Team"] = tableOne["Team"].map(teamMap)
        tableOne = tableOne.rename(columns={"Team":"team"})
        tableOne["season"] = i
        standings.append(tableOne)
        # avoid triggering hockey reference's rate limit (20 requests/minute) by waiting 5 seconds before re-entering the loop
        # it is only currently 17 requests needed but in the future with more season data it could end up being more so is important to plan ahead for this
        time.sleep(5)
    # get the standings in a csv file
    standings = pd.concat(standings, ignore_index=True)
    standings.to_csv("data/standings.csv", index=False)