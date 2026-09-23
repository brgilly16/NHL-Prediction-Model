import pandas as pd
def downloadGar():
    df = pd.read_csv("data/EH_gar_sk_stats_regular_2026-07-23.csv")
    yearsMap = {
    "10-11": 2010,
    "11-12": 2011,
    "12-13": 2012,
    "13-14": 2013,
    "14-15": 2014,
    "15-16": 2015,
    "16-17": 2016,
    "17-18": 2017,
    "18-19": 2018,
    "19-20": 2019,
    "20-21": 2020,
    "21-22": 2021,
    "22-23": 2022,
    "23-24": 2023,
    "24-25": 2024,
    "25-26": 2025
    }
    df["Season"] = df["Season"].map(yearsMap)
    # get GAR data in a csv file
    df.to_csv("data/gar.csv", index=False)