###################################################################
############ Setting up the workplace #############################
###################################################################
# Import packages
import os

import pandas as pd

import math

import numpy as np

import matplotlib.pyplot as plt

from astropy.table import Table

pd.set_option('mode.chained_assignment', None)
###################################################################
# This code requires Data Preparation script.
# Thus code contains a lot of replication, I will only explain
# first cases


''' Set the correct work directory '''

os.chdir('/Users/edgarakopyan/Desktop/Measuring Unfair Inequality/Data')

samplework = pd.read_pickle('Intermediarywork')

years = list(samplework.year.unique())

# Now we create the poverty thresholds
rosstat = pd.read_excel("data.xls")
rosstat = list(rosstat.iloc[4, 1:])
rosstat = pd.Series(rosstat, index = range(2001, 2019))

ymin = pd.Series( list([145397, 327000, 379000, 425000, 717, 963, 1133]),index = range(1994, 2001))
ymin = pd.concat([ymin, rosstat])


ymin = ymin[~ymin.index.isin([1997, 1999])]

ymin[ymin.index < 1998] = ymin[ymin.index < 1998]/1000

ymin = list(ymin)

# We also need  to adjust for inflation. I use IMF inflation data as opposed to Penn Table because the latter goes only
# until 2017.

inflation = pd.read_excel('imf-dm-export-20200516.xls')

inflation = inflation.iloc[1, 15:40]  # Select the valid years

inflation = inflation/100 + 1
# We have to multiply each consequentive element by the previous one to see how much price levels have changed over the
# period
for i in range(1, inflation.shape[0]):
    inflation.iloc[i] = inflation.iloc[i] * inflation.iloc[i-1]

finalinflation = inflation.iloc[inflation.shape[0]-1]

for i in range(0, inflation.shape[0]):
    inflation.iloc[i] = (inflation.iloc[i])/finalinflation

inflation = inflation.drop([1997, 1999])  # Since we do not have those years anyway

ymin = [x/y for x,y in zip(ymin, inflation)]

ymin = pd.Series(ymin, index= years)

types = list((samplework.type.value_counts()).index)

###################################################################
###################### Headcount Ratio ############################
###################################################################

HeadcountR = []

for j in years:
    a = samplework[(samplework.year == j) & (samplework.origsm == 1)]
    # Winsorise data
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    YMIN = ymin[j]
    lowincome = a[a.income <= YMIN]
    HeadcountR.append((lowincome.shape[0] / a.shape[0]) * 100)

WBheadcount = pd.read_excel("API_SI.POV.NAHC_DS2_en_excel_v2_1071128.xls")

WBheadcount = WBheadcount[(WBheadcount['Data Source'] == 'Russian Federation') | (WBheadcount['Data Source'] == \
                                                                                  'Country Name')]

WBheadcount = WBheadcount.iloc[1, 38: 63]

WBheadcount.index = range(1994, 2019)

WBheadcount = WBheadcount.drop(labels = [1997, 1999])

plt.scatter(years, HeadcountR, color="Red")

plt.scatter(years, list(WBheadcount), color="Blue")

plt.legend(["Author's Calculations", "World Bank Data"], mode="Expand")

plt.title("Headcount Ratio in Russia from 1994 to 2018", loc='center')

plt.xlabel("Year")

plt.ylabel("Headcount Ratio as a % of population")

plt.savefig("Headcount")

plt.clf()

###################################################################
#################### Poverty Gap Index ############################
###################################################################

PovertyGap = []

for j in years:
    a = samplework[(samplework.year == j) & (samplework.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    YMIN = ymin[j]
    lowincome = a[a.income <= YMIN]
    PovertyGap.append(((lowincome.shape[0]/a.shape[0]) - (1 / (a.shape[0] * YMIN)) * lowincome.income.sum())*100)

WBpoverty = pd.read_excel("API_SI.POV.UMIC.GP_DS2_en_excel_v2_1126782.xls")

WBpoverty  = WBpoverty [(WBpoverty ['Data Source'] == 'Russian Federation') | (WBpoverty ['Data Source'] == \
                                                                                  'Country Name')]

WBpoverty  = WBpoverty.iloc[1, 38: 63]

WBpoverty .index = range(1994, 2019)

WBpoverty  = WBpoverty .drop(labels=[1997, 1999])

# Now we plot the Poverty Gap
plt.plot(years, PovertyGap)

plt.plot(years, WBpoverty)

plt.legend(("Author's Calculations", "World Bank Data (Poverty gap at $5.50 a day 2011 PPP)"))

plt.title("Poverty Gap in Russia from 1994 to 2018", loc='center')

plt.xlabel("Year")

plt.ylabel("Poverty Gap")

plt.savefig("PovertyGap")

plt.clf()

###################################################################
########################## Watts Index ############################
###################################################################

Watts = []

def watts(dataframe, YMIN):
    b = dataframe[dataframe.income <= YMIN]
    WattsIndex = (1 / a.shape[0]) * (b.shape[0]) * math.log(YMIN) - (1 / a.shape[0]) * \
                (b.income.transform(math.log)).sum()
    return WattsIndex

for j in years:
    a = samplework[(samplework.year == j) & (samplework.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    YMIN = ymin[j]
    Watts.append(watts(a, YMIN = YMIN))

# Now we plot it and save the file

plt.plot(years, Watts)

plt.xlabel("Year")

plt.ylabel("Watts Index")

plt.title("Watts Index in Russia from 1994 to 2018", loc='center')

plt.savefig("Watts Index")

plt.clf()

# And save these values to use them in Pearson's correlation estimation in the next file
z = Table()

z['Watts'] = Watts

z['Poverty_Gap'] = PovertyGap

z['ymin'] = ymin

z.write("Intermediarylists", format='csv')
###################################################################
########################## GINI Index #############################
###################################################################

GINI = []

# Calculating the coefficient
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # Taken from Guest, Olivia (2017) Gini. [source code URL]: https://github.com/oliviaguest/gini
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1, array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

''' # This part is optional: 
    # I have manually checked if this code calculates the GINI coefficient by running the following for 2017 
a = samplework[(samplework.year == 2017) & (samplework.origsm == 1)]
a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
nonusedtypes = []
for i in types:
    if (a[a.type == i].shape[0] < 20) == True:
        nonusedtypes.append(i)
a = a[~a.type.isin(nonusedtypes)]
Ginimatrix = pd.DataFrame()
for i in range(0, a.shape[0]):
    Ginimatrix = pd.concat([Ginimatrix, pd.DataFrame(a.income)])
Ginimatrix.transpose()
Ginimatrix.shape
Ginisum = 0
for i in range(0, len(a.income)):
    for j in range(0, len(a.income)):
        Ginisum = Ginisum + abs(a.income.iloc[i] - a.income.iloc[j])
gini(np.array(a.income)) == (Ginisum) / (2 * ((a.shape[0]) ** 2) * a.income.mean())
'''
# Now on to calculating the Gini coefficient
for i in years:
    a = samplework[(samplework.year == i) & (samplework.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    GINI.append(gini(np.array(a.income)))
GINI = [x * 100 for x in GINI]
# Upload World Bank Inequality data
worldbank = pd.read_excel("API_SI.POV.GINI_DS2_en_excel_v2_1068874.xls")

WorldBankGini = worldbank[(worldbank['Data Source'] == 'Russian Federation') | (worldbank['Data Source'] == \
                                                                                'Country Name')]

WorldBankGini = WorldBankGini.iloc[1, 38: 63]

WorldBankGini.index = range(1994, 2019)

WorldBankGini = WorldBankGini.drop(labels = [1997, 1999])

# Now plot them both on one graph
plt.plot(years, GINI, color="Blue")

plt.plot(years, list(WorldBankGini), color="Red")

plt.title("Gini Coefficient in Russia from 1994 to 2018")

plt.xlabel("Year")

plt.ylabel("Gini Coefficient")

plt.legend(("Author's Calculations", "World Bank Data "))

plt.savefig("Gini")

plt.clf()
