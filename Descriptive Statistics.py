###################################################################
############ Setting up the workplace #############################
###################################################################
# Import packages
import time
import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn
###################################################################
start_time = time.clock()
os.chdir('/Users/edgarakopyan/Desktop/Measuring Unfair Inequality/Data')
samplework2 = pd.read_pickle('Intermediarywork')
ymin= 0
years = list(samplework2.year.unique())
types = list((samplework2.type.value_counts()).index)

###################################################################
################## Mean Log Deviation #############################
###################################################################

MLD = []
for j in years:
    a = samplework2[(samplework2.year == j) & (samplework2.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    ymin = a.income.median() * 0.6
    MLD.append(np.log(a.income.mean()) - (1 / a.shape[0]) * (np.log(a.income)).sum())

# Now plot ot on a graph
plt.plot(years, MLD)
plt.show()

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
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

# I have manually checked if this code calculates the GINI coefficient by running the following
a = samplework2[(samplework2.year == 2017) & (samplework2.origsm == 1)]
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

# Now on to calculating the Gini coefficient
GINI = []
for i in years:
    a = samplework2[(samplework2.year == i) & (samplework2.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    GINI.append(gini(np.array(a.income)))
GINI = [x * 100 for x in GINI]
# Upload World Bank Inequality data
worldbank = pd.read_excel("API_SI.POV.GINI_DS2_en_excel_v2_1068874.xls")
WorldBankGini = worldbank[(worldbank['Data Source'] == 'Russian Federation') | (worldbank['Data Source'] == 'Country Name')]
WorldBankGini = WorldBankGini.iloc[1, 38: 63]
WorldBankGini.index = range(1994, 2019)
WorldBankGini = WorldBankGini.drop(labels = [1997, 1999])

# Now plot them both on one graph
plt.plot(years, GINI, color= "Blue")
plt.plot(years, list(WorldBankGini), color = "Red")
plt.show()

###################################################################
###################### Headcount Ratio ############################
###################################################################

HeadcountR = []

for j in years:
    a = samplework2[(samplework2.year == j) & (samplework2.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    lowincome = a[a.income <= ymin]
    ymin = a.income.median() * 0.6
    HeadcountR.append(a.shape[0]/ lowincome.shape[0])

WBheadcount = pd.read_excel("API_SI.POV.NAHC_DS2_en_excel_v2_1071128.xls")
WBheadcount = WBheadcount[(WBheadcount['Data Source'] == 'Russian Federation') | (WBheadcount['Data Source'] == \
                                                                                  'Country Name')]
WBheadcount = WBheadcount.iloc[1, 38: 63]
WBheadcount.index = range(1994, 2019)
WBheadcount = WBheadcount.drop(labels = [1997, 1999])

plt.plot(years, HeadcountR, color = "Red")
plt.plot(years, list(WBheadcount), color = "Blue")
plt.legend(("Authors Calculations", "World Bank Data"), title = "Headcount Ratio:", mode = "Expand")
plt.show()

###################################################################
#################### Poverty Gap Index ############################
###################################################################

PovertyGap = []

for j in years:
    a = samplework2[(samplework2.year == j) & (samplework2.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    ymin = a.income.median() * 0.6
    lowincome = a[a.income <= ymin]
    PovertyGap.append((lowincome.shape[0]/a.shape[0]) - (1 / (a.shape[0] * ymin) ) * lowincome.income.sum())

WBpoverty = pd.read_excel("API_SI.POV.GAPS_DS2_en_excel_v2_1073352.xls")
WBpoverty  = WBpoverty [(WBpoverty ['Data Source'] == 'Russian Federation') | (WBpoverty ['Data Source'] == \
                                                                                  'Country Name')]
WBpoverty  = WBpoverty .iloc[1, 38: 63]
WBpoverty .index = range(1994, 2019)
WBpoverty  = WBpoverty .drop(labels = [1997, 1999])


plt.plot(years, PovertyGap)
plt.plot(years, WBpoverty)
plt.legend(("Authors Calculations", "World Bank Data"), title = "Poverty Raio:", mode = "Expand")
plt.show()

###################################################################
########################## Watts Index ############################
###################################################################

Watts = []

def watts(dataframe, ymin):
    b = dataframe[dataframe.income <= ymin]
    WattsIndex = (1 / a.shape[0]) * (b.shape[0]) * math.log(ymin) - (1 / a.shape[0]) * \
                (b.income.transform(math.log)).sum()
    return WattsIndex

for j in years:
    a = samplework2[(samplework2.year == j) & (samplework2.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    ymin = a.income.median() * 0.6
    Watts.append(watts(a, ymin= ymin))

plt.plot(years, Watts)
plt.show()

print(time.clock() - start_time, "seconds")


