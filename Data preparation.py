###################################################################
###########  Setting up the workplace #############################
###################################################################
import os

import numpy as np

import pandas as pd

import pyreadr

pd.set_option('mode.chained_assignment', None)

''' Set the correct work directory '''

os.getcwd()

os.chdir('/Users/edgarakopyan/Desktop/Measuring Unfair Inequality/Data')

# We need to read the RData file using pyreadr package
data = pyreadr.read_r('RLMS-HSE_IND_1994_2018_STATA.RData')

# And finally to get the dataframe we extract it from the created dictionary
workdata = data["table"]

# Let's make a copy of the dataset and work on the copy for now and for coding convenience we reset index
samplework = workdata.copy()

# And select the variables we want to work with
samplework = samplework[['J8', 'J38', 'J59', 'I4', 'H5', 'idind', 'J217A', 'J217B', 'J216AC08', 'J216BC08', \
                           'age','J60', 'year', 'ID_H', 'origsm', 'inwgt']]

###################################################################
###########  Cleaning parent occupation and education #############
###################################################################

# Parental education is present only in two years. Let's fill in parent education with the maximum value
# (both mother and father variables)

samplework['J217A'] = samplework.groupby('idind')['J217A'].transform(lambda x: x.fillna(x.max()))

samplework['J217B'] = samplework.groupby('idind')['J217B'].transform(lambda x: x.fillna(x.max()))

# Same should be done with parental occupation (both mother and father variables)
samplework['J216AC08'] = samplework.groupby('idind')['J216AC08'].transform(lambda x: x.fillna(x.max()))

samplework['J216BC08'] = samplework.groupby('idind')['J216BC08'].transform(lambda x: x.fillna(x.max()))

# Let's get rid of people not of working age
samplework = samplework[samplework.age.isin(range(25,60))]

# Now we need to remove rows with missing parent occupation and education.
# We remove both NA values as well as answers like don't know and refused to answer.
samplework = samplework[samplework['J216AC08'].notna()]

samplework = samplework[samplework['J216BC08'].notna()]

samplework = samplework[samplework['J217A'].notna()]

samplework = samplework[samplework['J217B'].notna()]

samplework = samplework[samplework['J216AC08'] != 99999997]

samplework = samplework[samplework['J216AC08'] != 99999996]

samplework = samplework[samplework['J216AC08'] != 99999995]

samplework = samplework[samplework['J216AC08'] != 99999998]

samplework = samplework[samplework['J216AC08'] != 99999994]

samplework = samplework[samplework['J216AC08'] != 99999999]

samplework = samplework[samplework['J216AC08'] != 99999993]

samplework = samplework[samplework['J216BC08'] != 99999997]

samplework = samplework[samplework['J216BC08'] != 99999996]

samplework = samplework[samplework['J216BC08'] != 99999995]

samplework = samplework[samplework['J216BC08'] != 99999998]

samplework = samplework[samplework['J216BC08'] != 99999999]

samplework = samplework[samplework['J216BC08'] != 99999994]

samplework = samplework[samplework['J216BC08'] != 99999993]

samplework = samplework[samplework['J217A'] != 99999999]

samplework = samplework[samplework['J217A'] != 99999996]

samplework = samplework[samplework['J217A'] != 99999993]

samplework = samplework[samplework['J217A'] != 99999998]

samplework = samplework[samplework['J217A'] != 99999997]

samplework = samplework[samplework['J217B'] != 99999999]

samplework = samplework[samplework['J217B'] != 99999998]

samplework = samplework[samplework['J217B'] != 99999997]

samplework = samplework[samplework['J217B'] != 99999996]

samplework = samplework[samplework['J217B'] != 99999993]

samplework = samplework.reset_index(drop=True)

# Then we clear the occupation variable by taking only the first digit  and structure it into 3 distinct groups

for i in range(0, samplework.shape[0]):
    while samplework.J216AC08[i] >= 10:
        samplework.J216AC08[i] = samplework.J216AC08[i] // 10

for i in range(0, samplework.shape[0]):
    while samplework.J216BC08[i] >= 10:
        samplework.J216BC08[i] = samplework.J216BC08[i] // 10

# And finally we create a new variable that divides occupation into 3 groups following the original Measuring
# Unfair Inequality paper

samplework['fatherwork'] = None

for i in range(0, samplework.shape[0]):
    if samplework['J216AC08'][i] in [1, 2, 3]:
        samplework['fatherwork'][i] = '2'
    elif samplework['J216AC08'][i] in [4, 5, 7, 8]:
        samplework['fatherwork'][i] = '1'
    else:
        samplework['fatherwork'][i] = '0'

samplework['fatherwork'].value_counts()

samplework['motherwork'] = None

for i in range(0, samplework.shape[0]):
    if samplework['J216BC08'][i] in [1, 2, 3]:
        samplework['motherwork'][i] = '2'
    elif samplework['J216BC08'][i] in [4, 5, 7, 8]:
        samplework['motherwork'][i] = '1'
    else:
        samplework['motherwork'][i] = '0'

# And now we create a variable of maximum parent occupation for each individual
samplework['maxparentwork'] = None

for i in range(0, samplework.shape[0]):
    samplework['maxparentwork'][i] = max(samplework['motherwork'][i], samplework['fatherwork'][i])

# Now we need to do the same with education variable

samplework['fathereducation'] = None

for i in range(0, samplework.shape[0]):
    if samplework['J217A'][i] in [9, 10, 11]:
        samplework['fathereducation'][i] = 2
    elif samplework['J217A'][i] in [4, 5, 6, 7, 8]:
        samplework['fathereducation'][i] = 1
    else:
        samplework['fathereducation'][i] = 0

samplework['fathereducation'].value_counts()

samplework['mothereducation'] = None

for i in range(0, samplework.shape[0]):
    if samplework['J217B'][i] in [9, 10, 11]:
        samplework['mothereducation'][i] = 2
    elif samplework['J217B'][i] in [4, 5, 6, 7, 8]:
        samplework['mothereducation'][i] = 1
    else:
        samplework['mothereducation'][i] = 0

# And now we create a variable of maximum parent education
samplework['maxparented'] = None

for i in range(0, samplework.shape[0]):
    samplework['maxparented'][i] = max(samplework['mothereducation'][i], samplework['fathereducation'][i])

###################################################################
###########  Cleaning national (ethnic)  background ###############
###################################################################

# Now on to national background
# Let's look at the share of missing values
len(samplework[samplework['I4'] == 99999997]) + len(samplework[samplework['I4'] == 99999998]) +\
    len(samplework[samplework['I4'] == 99999999]) + len(samplework[samplework['I4'] == np.nan])

# We have 436 missing values. Let's see how many slavic peoples we have
(len(samplework[samplework['I4'] == 2]) + len(samplework[samplework['I4'] == 1]) +\
    len(samplework[samplework['I4'] == 4]))/samplework.shape[0]
samplework.I4.value_counts()
# 89% of our dataset are slavic peoples
# As we can see, Slavs heavily dominate this variable and for many nationalities have only 1 observation. This means
# that it will be impossible to construct sufficiently big types to compare them to each other.
# However, we can divided into Slavic and non-slavic groups. We first get rid of missing observations and then divide.
samplework = samplework[(samplework.I4 != 99999997) & (samplework.I4 != 99999998) & (samplework.I4 != 99999999)]

samplework = samplework[samplework.I4 != np.nan]

# And now we divide into the two groups. I will bundle Slavs and Europeans into one group and the rest into another.
samplework['nationality'] = None

for i in list(samplework.index):
    if samplework['I4'][i] in [1, 2, 4, 5, 7, 10, 13]:
        samplework['nationality'][i] = 1
    else:
        samplework['nationality'][i] = 0

###################################################################
###########  Cleaning outcome variable ############################
###################################################################

# Let's look at the outcome variable. No sudden jumps in dont knows
samplework[samplework.J60 == 99999997].groupby(samplework.year).J60.value_counts() / samplework.groupby(\
    samplework.year).size()

# No sudden jumps in refusals.
samplework[samplework.J60 == 99999998].groupby(samplework.year).J60.value_counts() / samplework.groupby(\
    samplework.year).size()

# Relatively high in 1994 and 1995 but otherwise low in other years
samplework[samplework.J60 == 99999999].groupby(samplework.year).J60.value_counts() / samplework.groupby(\
    samplework.year).size()

# More 0 in 1996 and 1998 but otherwise also consistent
samplework[samplework.J60 == 0].groupby(samplework.year).J60.value_counts() / samplework.groupby(\
    samplework.year).size()

# Hence we can remove unnecessary values
samplework = samplework[~((samplework.J60 == 99999997.0) | (samplework.J60 == 99999998.0) | (samplework.J60 == \
                                                                                                99999999.0))]
# And we also need to remove observations where we have nonzero working hours but zero salary
samplework = samplework[~((samplework.J60 == 0) & ((samplework.J8 > 0) | (samplework.J38 > 0)| (samplework.J59 > 0)))]

samplework['income'] = samplework.J60
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

# Now we adjust the income by inflation
samplework = samplework.reset_index(drop=True)

for i in range(1,samplework.shape[0]):
    samplework.income[i] = samplework.income[i] / inflation[int(samplework.year[i])]

# We also have to take into account the redonimination of ruble in 1998 with the rate 1000 to 1
samplework.income[samplework.year < 1998] = samplework.income[samplework.year < 1998]/1000

# Finally replace 0 incomes with 1 to avoid problems when applying logarithm
samplework.income[samplework.income == 0] = 1

###################################################################
###########  Create types #########################################
###################################################################

# And we make types
samplework['type'] = samplework['maxparentwork'].astype('str') + '_' + samplework['maxparented'].astype('str') + \
  '_' + samplework['H5'].astype('str') + '_' + samplework['nationality'].astype('str')

types = list((samplework.type.value_counts()).index)

###################################################################
###########  Save dataset #########################################
###################################################################

# Save this file to be used in the next part
samplework.to_pickle('Intermediarywork')
