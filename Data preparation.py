###################################################################
###########  Setting up the workplace #############################
###################################################################
import os
import numpy as np
import pandas as pd
import pyreadr
import statsmodels.stats.weightstats
from statsmodels import zip
from samplics.weighting import SampleWeight

os.getcwd()
os.chdir('/Users/edgarakopyan/Desktop/Measuring Unfair Inequality/Data')
# We need to read the RData file using pyreadr package
data = pyreadr.read_r('RLMS-HSE_IND_1994_2018_STATA.RData')
# And finally to get the dataframe we extract it from the created dictionary
workdata = data["table"]
workdata.J60[workdata.J60 == np.nan] = 1
workdata.J60[workdata.J60 == 0] = 1
workdata["loginc"] = np.log(workdata.J60)
# Let's make a copy of the dataset and work on the copy for now and for coding convenience we reset index
samplework1 = workdata.copy()

# And select the variables we want to work with
samplework1 = samplework1[['J8', 'J38', 'J59', 'I3','I4', 'H5', 'idind', 'J217A', 'J217B', 'J216AC08', 'J216BC08', 'age',\
                           'J60', 'year',  'inwgt', 'ID_H', 'ID_W', 'origsm']]

###################################################################
###########  Cleaning parent occupation and education #############
###################################################################

# Filling in parent education

samplework1['J217A'] = samplework1.groupby('idind')['J217A'].transform(lambda x: x.fillna(x.max()))
samplework1['J217B'] = samplework1.groupby('idind')['J217B'].transform(lambda x: x.fillna(x.max()))

#Fill in parent occupation
samplework1['J216AC08'] = samplework1.groupby('idind')['J216AC08'].transform(lambda x: x.fillna(x.max()))
samplework1['J216BC08'] = samplework1.groupby('idind')['J216BC08'].transform(lambda x: x.fillna(x.max()))

# Let's get rid of people not of working age and have a look at sample attrition due to pension age
samplework1 = samplework1[samplework1.age.isin(range(25,60))]

# Now we need to remove rows with missing parent occupation and education. We remove both NA valeus as well as answers
# like don't know and refused to answer and
samplework2 = samplework1[samplework1['J216AC08'].notna()]

samplework2 = samplework2[samplework2['J216BC08'].notna()]

samplework2 = samplework2[samplework2['J217A'].notna()]

samplework2 = samplework2[samplework2['J217B'].notna()]

samplework2 = samplework2[samplework2['J216AC08']!= 99999997]

samplework2 = samplework2[samplework2['J216AC08']!= 99999996]

samplework2 = samplework2[samplework2['J216AC08']!= 99999995]

samplework2 = samplework2[samplework2['J216AC08']!= 99999998]

samplework2 = samplework2[samplework2['J216AC08']!= 99999994]

samplework2 = samplework2[samplework2['J216AC08']!= 99999999]

samplework2 = samplework2[samplework2['J216AC08']!= 99999993]

samplework2 = samplework2[samplework2['J216BC08']!= 99999997]

samplework2 = samplework2[samplework2['J216BC08']!= 99999996]

samplework2 = samplework2[samplework2['J216BC08']!= 99999995]

samplework2 = samplework2[samplework2['J216BC08']!= 99999998]

samplework2 = samplework2[samplework2['J216BC08']!= 99999999]

samplework2 = samplework2[samplework2['J216BC08']!= 99999994]

samplework2 = samplework2[samplework2['J216BC08']!= 99999993]

samplework2 = samplework2[samplework2['J217A']!= 99999999]

samplework2 = samplework2[samplework2['J217A']!= 99999996]

samplework2 = samplework2[samplework2['J217A']!= 99999993]

samplework2 = samplework2[samplework2['J217A']!= 99999998]

samplework2 = samplework2[samplework2['J217A']!= 99999997]

samplework2 = samplework2[samplework2['J217B']!= 99999999]

samplework2 = samplework2[samplework2['J217B']!= 99999998]

samplework2 = samplework2[samplework2['J217B']!= 99999997]

samplework2 = samplework2[samplework2['J217B']!= 99999996]

samplework2 = samplework2[samplework2['J217B']!= 99999993]

samplework2 = samplework2.reset_index(drop= True)

# Then we clear the occupation variable to structure into 3 distinct groups

for i in range(0, samplework2.shape[0]):
    while samplework2.J216AC08[i] >= 10:
        samplework2.J216AC08[i] = samplework2.J216AC08[i] // 10

for i in range(0, samplework2.shape[0]):
    while samplework2.J216BC08[i] >= 10:
        samplework2.J216BC08[i] = samplework2.J216BC08[i] // 10

list(samplework2['J216BC08'].value_counts().keys())

# And finally we create a new variable that divides occupation into 3 groups following the original Measuring
# Unfair Inequality paper

samplework2['fatherwork'] = None
for i in range(0, samplework2.shape[0]):
    if samplework2['J216AC08'][i] in [1, 2, 3]:
        samplework2['fatherwork'][i] = '2'
    elif samplework2['J216AC08'][i] in [4, 5, 7, 8]:
        samplework2['fatherwork'][i] = '1'
    else:
        samplework2['fatherwork'][i] = '0'
samplework2['fatherwork'].value_counts()

samplework2['motherwork'] = None
for i in range(0, samplework2.shape[0]):
    if samplework2['J216BC08'][i] in [1, 2, 3]:
        samplework2['motherwork'][i] = '2'
    elif samplework2['J216BC08'][i] in [4, 5, 7, 8]:
        samplework2['motherwork'][i] = '1'
    else:
        samplework2['motherwork'][i] = '0'

samplework2['motherwork'].value_counts()

# And now we create a variable of maximum parent occupation
samplework2['maxparentwork'] = None
for i in range(0, samplework2.shape[0]):
    samplework2['maxparentwork'][i] = max(samplework2['motherwork'][i], samplework2['fatherwork'][i])

# Now we need to do the same with education variable

samplework2['fathereducation'] = None
for i in range(0, samplework2.shape[0]):
    if samplework2['J217A'][i] in [9, 10, 11]:
        samplework2['fathereducation'][i] = 2
    elif samplework2['J217A'][i] in [4, 5, 6, 7, 8]:
        samplework2['fathereducation'][i] = 1
    else:
        samplework2['fathereducation'][i] = 0

samplework2['fathereducation'].value_counts()

samplework2['mothereducation'] = None
for i in range(0, samplework2.shape[0]):
    if samplework2['J217B'][i] in [9, 10, 11]:
        samplework2['mothereducation'][i] = 2
    elif samplework2['J217B'][i] in [4, 5, 6, 7, 8]:
        samplework2['mothereducation'][i] = 1
    else:
        samplework2['mothereducation'][i] = 0

samplework2['mothereducation'].value_counts()

# And now we create a variable of maximum parent education
samplework2['maxparented'] = None
for i in range(0, samplework2.shape[0]):
    samplework2['maxparented'][i] = max(samplework2['mothereducation'][i], samplework2['fathereducation'][i])

###################################################################
###########  Cleaning national (ethnic)  background ###############
###################################################################

# Now on to national background
# Let's look at the share of missing values
len(samplework2[samplework2['I4']==99999997]) + len(samplework2[samplework2['I4']==99999998]) +\
len(samplework2[samplework2['I4']==99999999]) + len(samplework2[samplework2['I4']== np.nan])
# We have 436 missing values. Let's see how many slavic peoples we have
shareofslavs = (len(samplework2[samplework2['I4']==2]) + len(samplework2[samplework2['I4']==1]) +\
len(samplework2[samplework2['I4']==4]))/samplework2.shape[0]
# 88% of our dataset are slavic peoples
# As we can see, Russians heavily dominate this variable and for many nationalities have only 1 observation. THis means
# that it will be impossible to construct sufficiently big types to compare them to each other.
# However, we can divided into Slavic and non-slavic groups. We first get rid of missing observations and then divide.
samplework2 = samplework2[(samplework2.I4!= 99999997) & (samplework2.I4!= 99999998) & (samplework2.I4!= 99999999)]
samplework2 = samplework2[samplework2.I4 != np.nan]
# And now we divide into the two groups. I will bundle Slavs and Europeans into one group and the rest into another.
samplework2['nationality'] = None
for i in list(samplework2.index):
    if samplework2['I4'][i] in [1, 2, 4, 5,7 , 10, 13 ]:
        samplework2['nationality'][i] = 1
    else:
        samplework2['nationality'][i] = 0

###################################################################
###########  Cleaning outcome variable ############################
###################################################################

# Let's look at the outcome variable. No suddent jumps in dont knows
samplework2[samplework2.J60 == 99999997].groupby(samplework2.year).J60.value_counts() / samplework2.groupby(samplework2.year).size()
# No sudden jumps in refusals.
samplework2[samplework2.J60 == 99999998].groupby(samplework2.year).J60.value_counts() / samplework2.groupby(samplework2.year).size()
# Relatively high in 1994 and 1995 but otherwise low in other years
samplework2[samplework2.J60 == 99999999].groupby(samplework2.year).J60.value_counts() / samplework2.groupby(samplework2.year).size()
# More 0 in 1996 and 1998 but otherwise also consistent
samplework2[samplework2.J60 == 0].groupby(samplework2.year).J60.value_counts() / samplework2.groupby(samplework2.year).size()


# Hence we can remove those values but retain zeros as those can be relevant for inequality calculations
samplework2 = samplework2[~((samplework2.J60 ==99999997.0) | (samplework2.J60 == 99999998.0) | (samplework2.J60 == 99999999.0))]
# And we also need to remove observations where we have nonzero working hours but zero salary
samplework2 = samplework2[~((samplework2.J60==0) & ((samplework2.J8>0) | (samplework2.J38>0)| (samplework2.J59>0)))]

samplework2['income'] = samplework2.J60
# We also need  to adjust for inflation. I use IMF inflation data as opposed to Penn Table because the latter goes only
# until 2017.

inflation = pd.read_excel('imf-dm-export-20200516.xls')
inflation = inflation.iloc[1, 15:40]
inflation = inflation/100 + 1
for i in range(1, inflation.shape[0]):
    inflation.iloc[i] = inflation.iloc[i] * inflation.iloc[i-1]
finalinflation = inflation.iloc[inflation.shape[0]-1]
for i in range(0, inflation.shape[0]):
    inflation.iloc[i] = (inflation.iloc[i])/finalinflation
inflation = inflation.drop([1997, 1999]) #Since we do not have those years anyway
# Now we adjust the income by inflation
samplework2 = samplework2.reset_index(drop=True)
for i in range(1,samplework2.shape[0]):
    samplework2.income[i] = samplework2.income[i] / inflation[int(samplework2.year[i])]

# We also have to take into account the redonimination of ruble in 1998 with the rate 1000 to 1
samplework2.income[samplework2.year < 1998] = samplework2.income[samplework2.year < 1998]/1000
# Finally replace 0 incomes with 1 to avoid problems when applying logarithm
samplework2.income[samplework2.income == 0] = 1

###################################################################
###########  Create types #########################################
###################################################################

# And we make types
samplework2['type'] = samplework2['maxparentwork'].astype('str') + '_' + samplework2['maxparented'].astype('str') + \
  '_' + samplework2['H5'].astype('str') + '_' + samplework2['nationality'].astype('str')
types = list((samplework2.type.value_counts()).index)

###################################################################
###########  Save dataset #########################################
###################################################################

# Save this file to be used in the next part
samplework2.to_pickle('Intermediarywork')


samplework3 = samplework2.copy()

aaa = samplework3[samplework3.year == 2017]
aaa.income.mean()
aaa['loginc'] = np.log(aaa.income)
d1 = statsmodels.stats.weightstats.DescrStatsW(aaa.income, weights=aaa.inwgt)
d2 = statsmodels.stats.weightstats.DescrStatsW(aaa.loginc, weights=aaa.inwgt)
aaa.income.median()
d1.quantile(0.5)
