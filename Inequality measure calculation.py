###################################################################
############  Setting up the workplace ############################
###################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import seaborn as sns
import scipy.stats

# Optional : check the time for the code to run
start_time = time.clock()

# Load the data file. Set the correct directory.
os.chdir('/Users/edgarakopyan/Desktop/Measuring Unfair Inequality/Data')
samplework2 = pd.read_pickle('Intermediarywork')

# Set variables for getting all inequality measures
years = list(samplework2.year.unique())
InequalityMeasure = []
UnfairInequalityShare = []
ymin= 0
types = list((samplework2.type.value_counts()).index)

###################################################################
########### Inequality Measure with Confidence Intervals ##########
###################################################################

a = a[(a.income > a.income.quantile(0.1)) & (a.income < a.income.quantile(0.9995))]

# Run the loop for calculating the inequality measure
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
    N = a.shape[0]
    # First part of unfair inequality estimation
    b = a[a.income <= ymin]  # the poor
    c = a[a.income > ymin]  # the rich
    firstpart = (1 / a.shape[0]) * (b.shape[0]) * math.log(ymin) - (1 / a.shape[0]) * \
                (b.income.transform(math.log)).sum() - \
                b.shape[0] / a.shape[0] + (1 / a.shape[0]) * (b.income / ymin).sum()
    # Now we calculate the second part
    a['ytilda'] = 1 - ymin / a.income
    # We create the variable for the Freedom from Poverty variable
    tFfP = (b.shape[0] * (ymin - b.income.mean())) / (c.shape[0] * (c.income.mean() - ymin))
    # And for equality of opportunity variable
    tEOp = pd.Series()
    for t in a.type.unique():
            tEOp[t] =   (a[a.type == t].income.mean() +\
                        (b[b.type == t].shape[0] / a[a.type == t].shape[0]) * (\
                        ymin - b[b.type == t].income.mean()) - tFfP * (\
                        c[c.type == t].shape[0] / a[a.type == t].shape[0]) * (\
                        c[c.type == t].income.mean() - ymin) -\
                        a.income.mean()) / (a[a.type == t].income.mean() + (\
                        b[b.type == t].shape[0] / a[a.type == t].shape[0]) * (\
                        ymin - b[b.type == t].income.mean()) \
                        - tFfP * (c[c.type == t].shape[0] / a[a.type == t].shape[0]) *\
                        (c[c.type == t].income.mean() - ymin) - ymin)
    # Now we prepare column for the second part
    a["secondpartcolumn"] = np.nan
    for i in a.index:
        a.secondpartcolumn[i] = 1 - a.ytilda[i] * (tFfP + tEOp[a.type[i]] - tEOp[a.type[i]] * tFfP)
    a.secondpartcolumn = np.log(a.secondpartcolumn)
    # And calculate the second part of inequality formula
    secondpart = (1 / N) * a[a.income > ymin].secondpartcolumn.sum()
    # Finally, we move on to the final part
    a["thirdpartcolumn"] = np.nan
    for i in a.index:
        a.thirdpartcolumn[i] = (a.ytilda[i] * (tFfP + tEOp[a.type[i]] - tEOp[a.type[i]] * tFfP)) / (1 - \
        a.ytilda[i] * (tFfP + tEOp[a.type[i]] -tEOp[a.type[i]] * tFfP))
    thirdpart = (1 / N) * a[a.income > ymin].thirdpartcolumn.sum()
    InequalityMeasure.append(firstpart + secondpart + thirdpart)
    UnfairInequalityShare.append((firstpart + secondpart + thirdpart) / (math.log(a.income.mean()) - (1 / a.shape[0]) *\
                                                                         (a.income.transform(math.log)).sum()))



##### Now let's get the 95% confidence intervals ######

matrix = pd.DataFrame(np.zeros((23, 0)))
for i in range(0,500):
    bootstrapeddata = samplework2.groupby(samplework2.year).apply(lambda x: x.sample(n= 500)).reset_index(drop=True)
    UnfairInequalityShareBoot = []
    for j in years:
        a = bootstrapeddata[(bootstrapeddata.year == j) & (bootstrapeddata.origsm == 1)]
        a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.995))]
        # Find groups with fewer than 20 people and remove them
        nonusedtypes = []
        for i in types:
            if (a[a.type == i].shape[0] < 20) == True:
                nonusedtypes.append(i)
        a = a[~a.type.isin(nonusedtypes)]
        ymin = a.income.median() * 0.6
        a = a.reset_index(drop=True)
        N = a.shape[0]
        # First part of unfair inequality estimation
        b = a[a.income <= ymin]  # the poor
        c = a[a.income > ymin]  # the rich
        firstpart = (1 / a.shape[0]) * (b.shape[0]) * math.log(ymin) - (1 / a.shape[0]) * \
                    (b.income.transform(math.log)).sum() - \
                    b.shape[0] / a.shape[0] + (1 / a.shape[0]) * (b.income / ymin).sum()
        # Now we calculate the second part
        a['ytilda'] = 1 - ymin / a.income
        # We create the variable for the Freedom from Poverty variable
        tFfP = (b.shape[0] * (ymin - b.income.mean())) / (c.shape[0] * (c.income.mean() - ymin))
        # And for equality of opportunity variable
        tEOp = pd.Series()
        for t in a.type.unique():
            tEOp[t] = (a[a.type == t].income.mean() + \
                       (b[b.type == t].shape[0] / a[a.type == t].shape[0]) * ( \
                                   ymin - b[b.type == t].income.mean()) - tFfP * ( \
                                   c[c.type == t].shape[0] / a[a.type == t].shape[0]) * ( \
                                   c[c.type == t].income.mean() - ymin) - \
                       a.income.mean()) / (a[a.type == t].income.mean() + ( \
                        b[b.type == t].shape[0] / a[a.type == t].shape[0]) * ( \
                                                       ymin - b[b.type == t].income.mean()) \
                                           - tFfP * (c[c.type == t].shape[0] / a[a.type == t].shape[0]) * \
                                           (c[c.type == t].income.mean() - ymin) - ymin)
        # Now we prepare column for the second part
        a["secondpartcolumn"] = np.nan
        for i in a.index:
            a.secondpartcolumn[i] = 1 - a.ytilda[i] * (tFfP + tEOp[a.type[i]] - tEOp[a.type[i]] * tFfP)
        a.secondpartcolumn = np.log(a.secondpartcolumn)
        # And calculate the second part of inequality formula
        secondpart = (1 / N) * a[a.income > ymin].secondpartcolumn.sum()
        # Finally, we move on to the final part
        a["thirdpartcolumn"] = np.nan
        for i in a.index:
            a.thirdpartcolumn[i] = (a.ytilda[i] * (tFfP + tEOp[a.type[i]] - tEOp[a.type[i]] * tFfP)) / (1 - \
                                                                                                        a.ytilda[i] * (\
                                                                                                        tFfP +tEOp[\
                                                                                                        a.type[i]] -\
                                                                                                        tEOp[a.type[\
                                                                                                        i]] * tFfP))
        thirdpart = (1 / N) * a[a.income > ymin].thirdpartcolumn.sum()
        UnfairInequalityShareBoot.append((firstpart + secondpart + thirdpart) / (math.log(a.income.mean()) -\
                                            (1 / a.shape[0]) * (a.income.transform(math.log)).sum()))
    UnfairInequalityShareBoot = pd.Series(UnfairInequalityShareBoot)
    matrix = pd.concat([matrix, UnfairInequalityShareBoot], axis=1)
    matrix.columns = range(0, len(matrix.columns))

bootstrapintmax = []
for i in range(0, matrix.shape[0]):
    bootstrapintmax.append(matrix.iloc[i,:].quantile(0.975))
bootstrapintmin = []
for i in range(0, matrix.shape[0]):
    bootstrapintmin.append(matrix.iloc[i,:].quantile(0.025))


####### Plot the Inequality Measure with confidence intervals #####

fig, ax = plt.subplots()
ax.plot(years,UnfairInequalityShare, color = "Red", linewidth=2.5)
ax.fill_between(years, bootstrapintmin,  bootstrapintmax, color='b', alpha=.1, linewidth=2.5)
plt.title("Unfair Inequality at % of Total Inequality with 95% confidence intervals", loc= 'center')
plt.xlabel("year")
plt.ylabel("Share of Total Inequality")
plt.show()

###################################################################
############ Get Lower and Upper Boundaries #######################
###################################################################

# Calculating only EOp
EOp = []
for j in years:
    a = samplework2[(samplework2.year == j)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    remainingtypes = list(a.type.unique())
    a['typemean'] = np.nan
    for i in list(a.index):
        a.typemean[i] = a[a.type == a.type[i]].income.mean()
    N = a.shape[0]
    EOp.append(np.log(a.income.mean()) - (1/N) * (np.log(a.typemean)).sum())

# Calculating only FfP
FfP = []
for j in years:
    a = samplework2[samplework2.year == j]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    ymin = a.income.median() * 0.6
    a = a.reset_index(drop=True)
    N = a.shape[0]
    # First part of the estimation
    b = a[a.income <= ymin]  # the poor
    c = a[a.income > ymin]  # the rich
    partone = (1 / a.shape[0]) * (b.shape[0]) * math.log(ymin) - (1 / a.shape[0]) * \
                (b.income.transform(math.log)).sum() - \
                b.shape[0] / a.shape[0] + (1 / a.shape[0]) * (b.income / ymin).sum()
    # Second part of the estimation
    a['ytilda'] = 1 - ymin / a.income
    # We create the variable for the Freedom from Poverty variable
    tFfP = (b.shape[0] * (ymin - b.income.mean())) / (c.shape[0] * (c.income.mean() - ymin))
    a["parttwo"] = np.nan
    for i in a.index:
        a.parttwo[i] = 1 - a.ytilda[i] * tFfP
    a.parttwo = np.log(a.parttwo)
    # And calculate the second part of inequality formula
    parttwo = (1 / N) * a[a.income > ymin].parttwo.sum()
    # Third part of the estimation
    a["partthree"] = np.nan
    for i in a.index:
        a.partthree[i] = (a.ytilda[i] * tFfP) / (1 - a.ytilda[i] * tFfP)
    partthree = (1 / N) * a[a.income > ymin].partthree.sum()
    # And finally we get the overall measure
    FfP.append(partone + parttwo + partthree)

UpperEOpbound = []
for i in range(0,len(EOp)):
    UpperEOpbound.append((EOp[i] / InequalityMeasure[i])* 100)
LowerEOpbound = []
for i in range(0,len(EOp)):
    LowerEOpbound.append(( (InequalityMeasure[i] - FfP[i]) / InequalityMeasure[i])* 100)

UpperFfPbound = []
for i in range(0,len(FfP)):
    UpperFfPbound.append((FfP[i] / InequalityMeasure[i])* 100)
LowerFfPbound = []
for i in range(0,len(FfP)):
    LowerFfPbound.append(((InequalityMeasure[i] - EOp[i]) / InequalityMeasure[i])* 100)

# Plot these on a graph

plt.plot(years, UpperFfPbound)
plt.plot(years, LowerFfPbound)
plt.show()
plt.plot(years, UpperEOpbound, color = "Blue")
plt.plot(years, LowerEOpbound, color = "Red")
plt.show()

# Check that my lower bounds do not joinly go over 100%

Totallow = tuple( x + y for x,y in zip(LowerFfPbound, LowerEOpbound))
plt.plot(years, Totallow)
plt.show()

print(time.clock() - start_time, "seconds")


