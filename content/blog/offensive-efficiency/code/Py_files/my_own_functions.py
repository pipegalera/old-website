import numpy as np

def offensive_efficiency(df):
    df["OE"] = (df.FG + df.AST) / (df.FGA - df.ORB + df.AST + df.TOV)
    df["OE"].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df

def EPS(df):
    df["EPS"] = np.sum(df["PTS"] * df["G"]) / (np.sum(df["OE"] * df["PTS"])) * df["OE"] * df["PTS"]
    return df

def raw_EOP(df):
    df["EOP"] = [(0.16 * df["AST"] + df["PTS"])] * [(df.FG + df.AST) / (df.FGA - df.ORB + df.AST + df.TOV)]
    df["EOP"].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df

def data_chart_lebron_melo(df):
    chart_df = df[(df["Player"] == "Carmelo Anthony") | (df["Player"] == "LeBron James")]
    chart_df = chart_df[["Player", "OE", "Season", "Tm"]].sort_values(by= ["Player", "Season"]).reset_index(drop = True)

    # Remove partial stats for Carmelo 2011 season, we take the total.
    chart_df = chart_df.drop(chart_df.index[[8,9]])
    chart_df = chart_df.round(2)

    return chart_df


#### Notes ####
"""
The formula of Offensive Efficiency can create infinity.
E.g, a player with 1 FG from 1 FGA and 1 ORB: (1 +0 )/ (1-1+0+0) = 1/0 = inf
For this marginal cases we have to replace infinity with 0.
"""
