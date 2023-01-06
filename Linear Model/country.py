import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

#Load the data
oecd_bli = pd.read_csv("bli_2017.csv", thousands=',')

gdp = pd.read_csv("gdp_2017.csv", thousands=',')


def prepare_country_stats(oecd_bli, gdp):
    loc1 = oecd_bli["LOCATION"]
    loc2 = gdp["LOCATION"]
    loc3 = loc1.isin(loc2)
    print(loc3)
    oecd_bli = oecd_bli[loc3]
    gdp = gdp[loc3]
    return oecd_bli, gdp




country_stats = prepare_country_stats(oecd_bli, gdp)
print(country_stats)

x = np.c_[country_stats["Value"]]
y = np.c_[country_stats["Value"]]

print(x)
print(y)
#Visualize the data
plt.scatter(x, y)
plt.show()
