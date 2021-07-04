# @Author: pipegalera
# @Date:   2021-05-20T17:27:54+02:00
# @Last modified by:   pipegalera
# @Last modified time: 2021-05-21T10:12:27+02:00



import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r"C:\Users\pipeg\Desktop\nba_analytics")

from Py_files.my_own_functions import *

################ Data Prep #####################################################

# Data
df_pg = pd.read_csv("./data/players_data_per_game.csv")
df_tot = pd.read_csv("./data/players_data_totals.csv")
df_team = pd.read_csv("./data/teams_data.csv")
df_adv = pd.read_csv("./data/players_data_advanced.csv")


################ Efficient Offensive  #####################################################

df = offensive_efficiency(df_pg)


# Table Efficient Offensive of the Top 20 Scorers
elite_2021 = df[df.Season == 2021].sort_values( by = "PTS", ascending = False).head(22).reset_index(drop = True).round(2)

# Remove Individual James Harden, taking only the average James Harden
elite_2021
elite_2021.drop([18, 21], inplace = True)
elite_2021.reset_index(drop = True, inplace = True)
elite_2021.index +=1

table_1 = elite_2021[["Player", "OE", "PTS"]].sort_values(by = "OE", ascending = False).reset_index(drop = True)
table_1.index +=1
print(table_1.to_markdown())



################  What is the mean OE of the season? ################
filter1 = df["Season"]==2021
filter2 = df["MP"]>(200/72)
players_2021 =df.where(filter1 & filter2).dropna()
players_2021["OE"].mean().round(2)


################  Scoring by position ################

scoring = df[df.Season == 2021].groupby(["Pos"]).sum()['PTS'].reset_index()
scoring['percent'] = (scoring["PTS"] / scoring["PTS"].sum() * 100).round(1)
scoring

# All the pos with center: 18.9, All the pos with point guard: 22,


################ Efficient Offensive Production ################################################

df = raw_EOP(df_pg)

table_2 = elite_2021[["Player", "EOP", "PTS"]].sort_values(by = "EOP", ascending = False).reset_index(drop = True)

table_2.index +=1

print(table_2.to_markdown())
