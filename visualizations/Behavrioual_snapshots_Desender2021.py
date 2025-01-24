#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:59:53 2024

@author: lorenzovanhoorde

Behavioural visualizations for model fitting
"""
### Import packages and define functiomns

import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import psychofit as psy 
except:
    import brainbox.behavior.psychofit as psy

def plot_psychometric(df, color='black', **kwargs):
    
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        ax = plt.gca()
    
    # from https://github.com/int-brain-lab/paper-behavior/blob/master/paper_behavior_functions.py#L391
    # summary stats - average psychfunc
    df2 = df.groupby(['sensory_evidence']).agg(count=('Response', 'count'),
                                               mean=('Response', 'mean')).reset_index()    
    # fit psychfunc
    pars, L = psy.mle_fit_psycho(df2.transpose().values,  # extract the data from the df
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [0, 2., 0.05, 0.05]),
                                 parmin=np.array(
                                     [df2['sensory_evidence'].min(), 0, 0., 0.]),
                                 parmax=np.array([df2['sensory_evidence'].max(), 4., 1, 1]))

    # plot psychfunc
    xrange = np.max(np.abs(df['sensory_evidence']))
    xlims = np.linspace(-xrange, xrange, num=100)
    sns.lineplot(x=xlims, y=psy.erf_psycho_2gammas(pars, xlims), 
                 color=color, zorder=10, **kwargs)
    
    # plot datapoints on top
    sns.lineplot(data=df, 
                 x='sensory_evidence', y='Response', err_style="bars", 
                 linewidth=0, mew=0.5, zorder=20,
                 marker='o', errorbar=('ci',68), color=color, **kwargs)
    
    # paramters in title
    ax.set_title(r'$\mu=%.2f, \sigma=%.2f, \gamma=%.2f, \lambda=%.2f$'%tuple(pars),
              fontsize='x-small')
    
# Create the 'block_breaks' variable per subject
def compute_block_breaks(subj_data):
    subj_data['block_breaks'] = ((subj_data.index + 1) % 60 == 0).astype(int)
    return subj_data
# =============================================================================================================
# =============================================================================================================

### DISREGARD
## Only for histogram comparing RT_dec & RTdecConf 
data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Motion discrimination/Desender_2021//data_Desender_2021_Cognit.csv')

# Plot to compare RTs in RT_dec & RT_decConf, should be run separate from actual pre-processing
data2 = data[data['RT_decConf'].notna()]  
data = data[data['RT_dec'].notna()] 
labels = ["Stimulus", "Stimulus + confidence"]

# Plot histograms with labels for the legend
plt.hist(data['RT_dec'], alpha=0.8, label=labels[0])  # First dataset
plt.hist(data2['RT_decConf'], alpha=0.8, label=labels[1])  # Second dataset

# Add labels and legend
plt.xlabel("Reaction Time (s)")
plt.xlim((0, 3))
plt.ylabel("Frequency")
plt.legend()  # Add legend to the plot
plt.show()

data['RT_dec'].mean() # 1.15
data2['RT_decConf'].mean() # 1.0 > similar timescale

# =============================================================================================================
# =============================================================================================================


## Load data
data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Motion discrimination/Desender_2021//data_Desender_2021_Cognit.csv')
data.head()
data.loc[data['RT_dec'] < 0, 'RT_dec'] # '-1 corresponds to, too slow responses
data.isna().sum() # NA's because RT_dec and RT_decConf are 2 different conditions.

# We decided to include both conditions > combine RT_dec & RTdecConf variable 
data['RT_dec_combined'] = data['RT_dec'].fillna(data['RT_decConf'])
data.isna().sum() # No NA's left in new variable
data['RT_dec'] = data['RT_dec_combined'] # For consistency in code / later wrapper function

# Some checks 
data['Stimulus'].unique() # Binary
any(data['RT_dec'] < 0) # Values under 0 > correspond to slow responses 
max(data['RT_dec']) # Max RT = 2.99 > time limit = max 3s
data['RT_dec'].mean() # Mean RT is 1.08

## Pre-processing

# Exclude training trials, 16200 rows / 30 subj. = 540 / 60 trials = 9 blocks
data = data[data['Training'] == 0] 
# Add variable that counts each trial per subject
data['trials.allN'] = data.groupby('Subj_idx').cumcount() + 1 # + 1 bc of Python indexing
# Add 'block_breaks', 60 trials per block
data['block_breaks'] = (data['trials.allN'] % 60 == 0).astype(int)
# Define sensory evidence based on coherence and side
data['sensory_evidence'] = np.where(data['Stimulus'] == 1, data['Coherence'], -data['Coherence'])
# Accuracy variable
data['Accuracy'] = (data['Stimulus'] == data['Response']).astype(int) # astype(int), converts true to 1 and false to 0
# convert too slow responses from -1 t0 -0.25 for aesthetic purpose
data.loc[data['RT_dec'] == -1, 'RT_dec'] = -0.25

# Because we combine two conditions I want a var. that indicates when the condition switches for the plot

# Create condition switch variable
data['cond_switch'] = 0 # initialize
data['cond_switch'] = data['Condition'] != data['Condition'].shift(1)

# Reset 'cond_switch' to 0 whenever 'Subj_idx' changes
data['cond_switch'] = data['cond_switch'] * (data['Subj_idx'] == data['Subj_idx'].shift(1)).astype(int)



# ================================================================================================================

### Visualize datasets

## Behavioural snapshots - average (different than in other scripts bc 'cond_switch' var.)

# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1, 1, 2], figsize=(20, 5))

# 1. Psychometric
plot_psychometric(data, ax=ax[0])
ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', ylim=[-0.05, 1.05])

# 2. Chronometric
sns.lineplot(
    data=data, ax=ax[1],
    x='sensory_evidence', y='RT_dec', err_style='bars',
    linewidth=1, estimator=np.median,
    mew=0.5, marker='o', errorbar=('ci', 68), color='black'
)
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', ylim=[0.7, 1.3])

# 3. Plot session
sns.scatterplot(
    data=data[data['Subj_idx'] == 1], ax=ax[2],
    x='trials.allN', y='RT_dec',
    style='Accuracy', hue='Accuracy',
    palette={1.: "#009E73", 0.: "#D55E00"},
    markers={1.: 'o', 0.: 'X'}, s=10, edgecolors='face',
    alpha=0.5, legend=False
)
ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

# Compute running median for filtered data
subject_data = data[data['Subj_idx'] == 1][['trials.allN', 'RT_dec']]
running_median = subject_data.rolling(10, on='trials.allN').median()

# Plot the running median line on the session subplot
sns.lineplot(
    data=running_median, ax=ax[2],
    x='trials.allN', y='RT_dec', color='black', errorbar=None
)

# Add vertical lines to mark block breaks
if 'block_breaks' in data.columns:
    subj_block_breaks = data[(data['block_breaks'] == 1) & (data['Subj_idx'] == 1)]
    block_end_trials = subj_block_breaks['trials.allN'].unique()  # Get unique trial numbers
    for x in block_end_trials:
        ax[2].axvline(x=x, color='blue', alpha=0.2, linestyle='--')

# Add red vertical lines for condition switches
cond_switch_trials = data[(data['cond_switch'] == 1) & (data['Subj_idx'] == 1)]['trials.allN']  # Filter for the specific subject
for trial in cond_switch_trials:
    ax[2].axvline(x=trial, color='red', alpha=0.8, linestyle='--', linewidth=1)

plt.show()


# ================================================================================================================

## Behavioural snapshots - per participant
random_subjects = np.random.choice(data['Subj_idx'].unique(), size=15, replace=False)

# Loop over the randomly selected subjects
for subj in random_subjects:
    # Filter data for the current subject
    subj_data = data[data['Subj_idx'] == subj]
    
    # Compute accuracy for the subject
    accuracy = subj_data['Accuracy'].mean()

    # Set up figure and axes
    fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1, 1, 2], figsize=(20, 5))

    # 1. Psychometric
    plot_psychometric(subj_data, ax=ax[0])  # Adjust plot_psychometric to work per subject
    ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', ylim=[-0.05, 1.05])

    # 2. Chronometric
    sns.lineplot(
        data=subj_data, ax=ax[1],
        x='sensory_evidence', y='RT_dec', err_style='bars',
        linewidth=1, estimator=np.median,
        mew=0.5, marker='o', errorbar=('ci', 68), color='black'
    )
    ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)')

    # 3. Plot session
    sns.scatterplot(
        data=subj_data, ax=ax[2],
        x='trials.allN', y='RT_dec',
        style='Accuracy', hue='Accuracy',
        palette={1.: "#009E73", 0.: "#D55E00"},
        markers={1.: 'o', 0.: 'X'}, s=10, edgecolors='face',
        alpha=0.5, legend=False
    )
    ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

    # Compute running median for filtered data
    running_median = subj_data[['trials.allN', 'RT_dec']].rolling(10, on='trials.allN').median()

    # Plot the running median line on the session subplot
    sns.lineplot(
        data=running_median, ax=ax[2],
        x='trials.allN', y='RT_dec', color='black', errorbar=None
    )

    # Add vertical lines to mark block breaks
    if 'block_breaks' in data.columns:
        subj_block_breaks = subj_data[subj_data['block_breaks'] == 1]
        block_end_trials = subj_block_breaks['trials.allN'].unique()  # Get unique trial numbers
        for x in block_end_trials:
            ax[2].axvline(x=x, color='blue', alpha=0.2, linestyle='--')

    # Add red vertical lines for condition switches
    cond_switch_trials = subj_data[subj_data['cond_switch'] == 1]['trials.allN']
    for x in cond_switch_trials:
        ax[2].axvline(x=x, color='red', alpha=0.8, linestyle='--')

    # Display the plot
    plt.suptitle(f'Subject {subj} | Accuracy: {accuracy:.2%}')
    plt.show()

# ==============================================================================================================

### Length of experiment 

## Historgam of number of trials 

# Calculate the number of trials per subject
trials_per_subject = data['trials.allN'].groupby(data['Subj_idx']).count()

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(trials_per_subject, bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Trials')
plt.ylabel('Number of Subjects')
plt.title('Histogram of Number of Trials Per Subject')
plt.suptitle('n = 540 for all subjects!')
plt.show()

## Trial length
data['RT_dec'].mean() # Mean RT
data['RT_dec'].groupby(data['Subj_idx']).mean() # Mean RT per subject
total_time_per_subject = data.groupby('Subj_idx')['RT_dec'].sum() #total time spent in the experiment per subject
total_time_per_subject

print("Mean time spent in experiment:", total_time_per_subject.mean())
print("Minimum total time:", total_time_per_subject.min())
print("Maximum total time:", total_time_per_subject.max())


# ==============================================================================================================
## Data artefacts 

# All -1 values, have no responses > too slow!
huh = data[data['RT_dec'] < 0] 
all(huh['Response'].isna()) # True 
print(data.isna().sum()) # But there are 275 -1 values in RT_dec
## Just realised this is because you have immediate and delayed condition 
## >> with RT and conf being selected at the same time in immediate condition
# To confirm NA's
No_response = data[data['Response'].isna()] 
No_response[['RT_dec', 'RT_conf', 'RT_decConf']] # All no responses 

## Check whether RT is affected by simultaneous reporting > to decide on including these trials or not 
print(data['RT_dec'].mean(), data['RT_conf'].mean(), data['RT_decConf'].mean()) # RT doesn't seem really affected by 
# reporting simultaneously. Even faster while reporting both (beware of effect -1 tho)
# RT conf mean is negative because of -99 values??? Whilst there are also -1 values. Weird, but we don´t use it anyway

# ================================================================================================================
### Plot RT_DecConf

## Pre-processing

# Define signed contrast based on coherence and side
data2['sensory_evidence'] = np.where(data2['Stimulus'] == 1, data2['Coherence'], -data2['Coherence']) # Assume that 1 is right here
# Add variable that counts each trial per subject
data2['trials.allN'] = data2.groupby('Subj_idx').cumcount() + 1 # + 1 bc of Python indexing
# Accuracy variable
data2['Accuracy'] = (data2['Stimulus'] == data2['Response']).astype(int) # astype(int), converts true to 1 and false to 0
# convert too slow responses from -1 t0 -0.25 for aesthetic purpose
data2.loc[data2['RT_decConf'] == -1, 'RT_decConf'] = -0.25

## Plot
# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1,1,2], figsize=(20,5))

# 1. psychometric
plot_psychometric(data2, ax=ax[0])
ax[0].set(xlabel='Sesnory evidence', ylabel='Choice (fraction)',
        ylim=[-0.05, 1.05])

# Chronometric 
sns.lineplot(data = data2, ax=ax[1],
             x= 'sensory_evidence', y = 'RT_decConf', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', 
               ylim=[0, 1.6])

# Plot session
sns.scatterplot(data = data2[data2['Subj_idx'] == 1], ax=ax[2],
                x='trials.allN', y='RT_decConf',                  
                 style='Accuracy', hue='Accuracy',
                 palette={1.:"#009E73", 0.:"#D55E00"}, 
                 markers={1.:'o', 0.:'X'}, s=10, edgecolors='face',
                 alpha=.5, legend=False)
ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

# Compute running median for filtered data
subject_data = data2[data2['Subj_idx'] == 1][['trials.allN', 'RT_decConf']]
running_median = subject_data.rolling(10, on='trials.allN').median()

# Plot the running median line on the second subplot
sns.lineplot(
    data=running_median, ax=ax[2],
    x='trials.allN', y='RT_decConf', color='black', errorbar=None
)

# TODO: Add blocks > think they have this in the data 
# add lines to mark breaks, if these were included
if 'block_breaks' in data.columns:
    if data['block_breaks'].iloc[0] == 'y':
        [ax[1].axvline(x, color='blue', alpha=0.2, linestyle='--') 
             for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
plt.show()

### ======================================================================================================

# New plotting code to add condition background 

## Globally

# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1, 1, 2], figsize=(20, 5))

# 1. Psychometric
plot_psychometric(data, ax=ax[0])
ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', ylim=[-0.05, 1.05])

# 2. Chronometric
sns.lineplot(
    data=data, ax=ax[1],
    x='sensory_evidence', y='RT_dec', err_style='bars',
    linewidth=1, estimator=np.median,
    mew=0.5, marker='o', errorbar=('ci', 68), color='black'
)
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', ylim=[0.7, 1.3])

# 3. Plot session
sns.scatterplot(
    data=data[data['Subj_idx'] == 1], ax=ax[2],
    x='trials.allN', y='RT_dec',
    style='Accuracy', hue='Accuracy',
    palette={1.: "#009E73", 0.: "#D55E00"},
    markers={1.: 'o', 0.: 'X'}, s=10, edgecolors='face',
    alpha=1, legend=False
)
ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

# Change background color dynamically based on 'Condition' for each trial
for trial_idx in range(len(data)):
    condition = data['Condition'].iloc[trial_idx]  # Get the condition for each trial
    trial_number = data['trials.allN'].iloc[trial_idx]  # Get the trial number
    
    if condition == 0:
        ax[2].axvspan(trial_number - 0.5, trial_number + 0.5, color='lightblue', alpha=0.5)
    elif condition == 1:
        ax[2].axvspan(trial_number - 0.5, trial_number + 0.5, color='lightgreen', alpha=0.5)
    else:
        ax[2].axvspan(trial_number - 0.5, trial_number + 0.5, color='lightgray', alpha=0.5)

# Compute running median for filtered data
subject_data = data[data['Subj_idx'] == 1][['trials.allN', 'RT_dec']]
running_median = subject_data.rolling(10, on='trials.allN').median()

# Plot the running median line on the session subplot
sns.lineplot(
    data=running_median, ax=ax[2],
    x='trials.allN', y='RT_dec', color='black', errorbar=None
)

# Add vertical lines to mark block breaks
if 'block_breaks' in data.columns:
    subj_block_breaks = data[(data['block_breaks'] == 1) & (data['Subj_idx'] == 1)]
    block_end_trials = subj_block_breaks['trials.allN'].unique()  # Get unique trial numbers
    for x in block_end_trials:
        ax[2].axvline(x=x, color='blue', alpha=0.2, linestyle='--')

# Add red vertical lines for condition switches
cond_switch_trials = data[(data['cond_switch'] == 1) & (data['Subj_idx'] == 1)]['trials.allN']  # Filter for the specific subject
for trial in cond_switch_trials:
    ax[2].axvline(x=trial, color='red', alpha=0.8, linestyle='--', linewidth=1)

plt.show()

### ======================================================================================================

## Per participant

## Behavioural snapshots - per participant
random_subjects = np.random.choice(data['Subj_idx'].unique(), size=15, replace=False)

# Loop over the randomly selected subjects
for subj in random_subjects:
    # Filter data for the current subject
    subj_data = data[data['Subj_idx'] == subj]
    
    # Compute accuracy for the subject
    accuracy = subj_data['Accuracy'].mean()

    # Set up figure and axes
    fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1, 1, 2], figsize=(20, 5))

    # 1. Psychometric
    plot_psychometric(subj_data, ax=ax[0])  # Adjust plot_psychometric to work per subject
    ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', ylim=[-0.05, 1.05])

    # 2. Chronometric
    sns.lineplot(
        data=subj_data, ax=ax[1],
        x='sensory_evidence', y='RT_dec', err_style='bars',
        linewidth=1, estimator=np.median,
        mew=0.5, marker='o', errorbar=('ci', 68), color='black'
    )
    ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)')

    # 3. Plot session
    sns.scatterplot(
        data=subj_data, ax=ax[2],
        x='trials.allN', y='RT_dec',
        style='Accuracy', hue='Accuracy',
        palette={1.: "#009E73", 0.: "#D55E00"},
        markers={1.: 'o', 0.: 'X'}, s=10, edgecolors='face',
        alpha=0.5, legend=False
    )
    ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

    # Change background color dynamically based on 'Condition' for each trial
    for trial_idx in range(len(subj_data)):
        condition = subj_data['Condition'].iloc[trial_idx]  # Get the condition for each trial
        trial_number = subj_data['trials.allN'].iloc[trial_idx]  # Get the trial number
        
        if condition == 0:
            ax[2].axvspan(trial_number - 0.5, trial_number + 0.5, color='lightblue', alpha=0.5)
        elif condition == 1:
            ax[2].axvspan(trial_number - 0.5, trial_number + 0.5, color='lightgreen', alpha=0.5)
        else:
            ax[2].axvspan(trial_number - 0.5, trial_number + 0.5, color='lightgray', alpha=0.5)

    # Compute running median for filtered data
    running_median = subj_data[['trials.allN', 'RT_dec']].rolling(10, on='trials.allN').median()

    # Plot the running median line on the session subplot
    sns.lineplot(
        data=running_median, ax=ax[2],
        x='trials.allN', y='RT_dec', color='black', errorbar=None
    )

    # Add vertical lines to mark block breaks
    if 'block_breaks' in data.columns:
        subj_block_breaks = subj_data[subj_data['block_breaks'] == 1]
        block_end_trials = subj_block_breaks['trials.allN'].unique()  # Get unique trial numbers
        for x in block_end_trials:
            ax[2].axvline(x=x, color='blue', alpha=0.2, linestyle='--')

    # Add red vertical lines for condition switches
    cond_switch_trials = subj_data[subj_data['cond_switch'] == 1]['trials.allN']
    for x in cond_switch_trials:
        ax[2].axvline(x=x, color='red', alpha=0.8, linestyle='--')

    # Display the plot
    plt.suptitle(f'Subject {subj} | Accuracy: {accuracy:.2%}')
    plt.show()
