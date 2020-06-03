###################################################################
############  Setting up the workplace ############################
###################################################################
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

pd.set_option('mode.chained_assignment', None)

# This code requires Data Preparation and Descriptive statistics script. Load the data file. Set the correct directory.

''' Set the correct work directory '''

os.chdir('/Users/edgarakopyan/Desktop/Measuring Unfair Inequality/Data')

samplework = pd.read_pickle('Intermediarywork')

Lists = pd.read_csv("Intermediarylists")

Watts = list(Lists.iloc[:, 0])

PovertyGap = list(Lists.iloc[:, 1])


# Set variables for getting all inequality measures

years = list(samplework.year.unique())

InequalityMeasure = []

UnfairInequalityShare = []

ymin = 0

types = list((samplework.type.value_counts()).index)

###################################################################
################## Mean Log Deviation #############################
###################################################################

MLD = []

for j in years:
    a = samplework[(samplework.year == j) & (samplework.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    ymin = a.income.median() * 0.6
    MLD.append(np.log(a.income.mean()) - (1 / a.shape[0]) * (np.log(a.income)).sum())


###################################################################
########### Inequality Measure with Confidence Intervals ##########
###################################################################

# Run the loop for calculating the inequality measure

for j in years:
    a = samplework[(samplework.year == j) & (samplework.origsm == 1)]
    a = a[(a.income > a.income.quantile(0.01)) & (a.income < a.income.quantile(0.9995))]
    # Find groups with fewer than 20 people and remove them
    nonusedtypes = []
    for i in types:
        if (a[a.type == i].shape[0] < 20) == True:
            nonusedtypes.append(i)
    a = a[~a.type.isin(nonusedtypes)]
    ymin = a.income.median() * 0.6
    N = a.shape[0]
    # I break up the estimation into three parts, based on the three summations
    # in the original equation. First part of unfair inequality estimation
    b = a[a.income <= ymin]  # the poor
    c = a[a.income > ymin]  # the rich
    firstpart = (1 / a.shape[0]) * (b.shape[0]) * math.log(ymin) - (1 / a.shape[0]) * \
                (b.income.transform(math.log)).sum() - \
                b.shape[0] / a.shape[0] + (1 / a.shape[0]) * (b.income / ymin).sum()
    # Now we calculate the second part
    a['ytilda'] = 1 - ymin / a.income
    # We create the variable for the Freedom from Poverty variable
    tFfP = (b.shape[0] * (ymin - b.income.mean())) / (c.shape[0] * (c.income.mean() - ymin))
    # And for equality of opportunity variable for each type
    tEOp = pd.Series()
    for t in a.type.unique():
            tEOp[t] = (a[a.type == t].income.mean() +\
                        (b[b.type == t].shape[0] / a[a.type == t].shape[0]) * (\
                        ymin - b[b.type == t].income.mean()) - tFfP * (\
                        c[c.type == t].shape[0] / a[a.type == t].shape[0]) * (\
                        c[c.type == t].income.mean() - ymin) -\
                        a.income.mean()) / (a[a.type == t].income.mean() + (\
                        b[b.type == t].shape[0] / a[a.type == t].shape[0]) * (\
                        ymin - b[b.type == t].income.mean()) \
                        - tFfP * (c[c.type == t].shape[0] / a[a.type == t].shape[0]) *\
                        (c[c.type == t].income.mean() - ymin) - ymin)
    # Now we prepare column for the second part of the equation
    a["secondpartcolumn"] = np.nan
    for i in a.index:
        a.secondpartcolumn[i] = 1 - a.ytilda[i] * (tFfP + tEOp[a.type[i]] - tEOp[a.type[i]] * tFfP)
    a.secondpartcolumn = np.log(a.secondpartcolumn) # During this log transformation an error will show up
    # This is because in the original equation we should sum over rich people only (who will have positive values in
    # the second part column) but the command constructs a column over all people both rich and poor.
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
    bootstrapeddata = samplework.groupby(samplework.year).apply(lambda x: x.sample(n=500)).reset_index(drop=True)
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
    bootstrapintmax.append(matrix.iloc[i, :].quantile(0.975))

bootstrapintmin = []

for i in range(0, matrix.shape[0]):
    bootstrapintmin.append(matrix.iloc[i, :].quantile(0.025))


####### Plot the Inequality Measure with confidence intervals #####

fig, ax = plt.subplots()

ax.plot(years, UnfairInequalityShare, color="Red", linewidth=2.5)

ax.fill_between(years, bootstrapintmin,  bootstrapintmax, color='Red', alpha=.1, linewidth=2.5)

plt.title("Unfair Inequality at % of Total Inequality and Mean Log Deviation", loc='center')

plt.xlabel("Year")

ax.set_ylabel("Share of Total Inequality")

ax.tick_params(axis='y', labelcolor='Red')

ax2 = ax.twinx()

ax2.set_ylabel('Mean Log Deviation')

ax2.plot(years, MLD, color="Green")

ax2.tick_params(axis='y', labelcolor='Green')

plt.savefig("Unfair")

plt.clf()

###################################################################
############ Get Lower and Upper Boundaries #######################
###################################################################

# Calculating only EOp
EOp = []

for j in years:
    a = samplework[(samplework.year==j) & (samplework.origsm==1)]
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
    a = samplework[(samplework.year == j) & (samplework.origsm == 1)]
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

for i in range(0, len(EOp)):
    LowerEOpbound.append(( (InequalityMeasure[i] - FfP[i]) / InequalityMeasure[i]) * 100)

UpperFfPbound = []

for i in range(0, len(FfP)):
    UpperFfPbound.append((FfP[i] / InequalityMeasure[i]) * 100)

LowerFfPbound = []

for i in range(0, len(FfP)):
    LowerFfPbound.append(((InequalityMeasure[i] - EOp[i]) / InequalityMeasure[i]) * 100)

# Plot these on a graph

line1, = plt.plot(years, UpperFfPbound, label = 'Freedom from Poverty Bounds', color="Red")

plt.plot(years, LowerFfPbound, color="Red")

plt.title("Bounds of Freedom from Poverty and Equality of Opportunity")

plt.xlabel("Year")

plt.ylabel("Share of Total Unfair Inequality")

first_label = plt.legend(handles=[line1], loc='upper right')

plt.gca().add_artist(first_label)

line2, = plt.plot(years, UpperEOpbound, label='Equality of Opportunity Bounds', color="Blue")

plt.plot(years, LowerEOpbound, color="Blue")

second_label = plt.legend(handles=[line2], loc='lower right')

plt.savefig("Decomposition")

plt.clf()

###################################################################
############ Removing unnecesary files ############################
###################################################################

os.remove("Intermediarylists")

os.remove("Intermediarywork")


