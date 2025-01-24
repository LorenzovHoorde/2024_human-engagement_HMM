#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:33:24 2024

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

# =============================================================================================================
## Task 1

# ================================================================================================================

data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Orientation discrimination/Rahnev_2013/data_Rahnev_2013.csv')
data.head()
data.columns

## The order of the conditions is counterbalanced across subjects, maybe use 1st condition for everyone?
## Should al be pre-TMS, so no stimulation-induced difference in brain activity 
# Extract the first condition for each subject

# Add variable that counts each trial per subject
data['trials.allN'] = data.groupby('Subj_idx').cumcount() + 1 # + 1 bc of Python indexing

first_condition_data = (
    data
    .sort_values(['Subj_idx', 'trials.allN'])  # Ensure data is ordered by trial number
    .groupby('Subj_idx')  # Group by each subject
    .apply(lambda group: group[group['Condition'] == group['Condition'].iloc[0]])  # Filter to keep only first condition
    .reset_index(drop=True)  # Reset index after filtering
)

# Verify the data contains only the first condition per subject
print(first_condition_data['Condition'].value_counts()) #  1, 3 & 5 are indeed all pre-TMS 
first_condition_data.groupby(first_condition_data['Subj_idx'])['Condition'].unique() # All subj have 1 cond.
data.groupby(data['Subj_idx'])['Condition'].unique() # Instead of 6
data = first_condition_data # Edit dataframe

data['Contrast'].unique()
len(data['Contrast'].unique()) # 3 unique contrast levels 


data['Stimulus'].unique() # Coded as 1 and 2 here
any(data['RT_decConf'] < 0) # No RT values under 0
max(data['RT_decConf']) # Max RT = 1.797 > response window of 1.8s
data['RT_decConf'].mean() # Mean RT is 0.73
any(data['RT_decConf'].isna()) # NA's > there was a time limit here 

## Pre-processing

# Convert Stimulus and Response into binary variable (for psychometric function)
data['Stimulus'] = data['Stimulus'].replace({1: 0, 2: 1})
data['Response'] = data['Response'].replace({1: 0, 2: 1})

# Define sensory evidence based on coherence and side
data['sensory_evidence'] = np.where(data['Stimulus'] == 1, data['Contrast'], -data['Contrast'])
# Accuracy variable
data['Accuracy'] = (data['Stimulus'] == data['Response']).astype(int) # astype(int), converts true to 1 and false to
# Add 'block_breaks', 140 trials per block
data['block_breaks'] = (data['trials.allN'] % 140 == 0).astype(int)

# ================================================================================================================

### Visualize datasets

## Behavioural snapshots - average


fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1,1,2], figsize=(20,5))

# 1. psychometric
plot_psychometric(data, ax=ax[0])
ax[0].set(xlabel='Sesnory evidence', ylabel='Choice (fraction)',
        ylim=[-0.05, 1.05])

# Chronometric 
sns.lineplot(data = data, ax=ax[1],
             x= 'sensory_evidence', y = 'RT_decConf', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)')

# Plot session
sns.scatterplot(data = data[data['Subj_idx'] == 1], ax=ax[2],
                x='trials.allN', y='RT_decConf',                  
                 style='Accuracy', hue='Accuracy',
                 palette={1.:"#009E73", 0.:"#D55E00"}, 
                 markers={1.:'o', 0.:'X'}, s=10, edgecolors='face',
                 alpha=.5, legend=False)
ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

# Compute running median for filtered data
subject_data = data[data['Subj_idx'] == 1][['trials.allN', 'RT_decConf']]
running_median = subject_data.rolling(10, on='trials.allN').median()

# Plot the running median line on the second subplot
sns.lineplot(
    data=running_median, ax=ax[2],
    x='trials.allN', y='RT_decConf', color='black', errorbar=None
)

# Add vertical lines to mark block breaks
# Filter block breaks for the specific subject
if 'block_breaks' in data.columns:
    subj_block_breaks = data[(data['block_breaks'] == 1) & (data['Subj_idx'] == 1)]
    block_end_trials = subj_block_breaks['trials.allN'].unique()  # Get unique trial numbers
    for x in block_end_trials:
        ax[2].axvline(x=x, color='blue', alpha=0.2, linestyle='--')


plt.show()

# ================================================================================================================

# ================================================================================================================

## Behavioural snapshots - per participant

#  Select all 12 subjects
random_subjects = np.random.choice(data['Subj_idx'].unique(), size=12, replace=False)

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
        x='sensory_evidence', y='RT_decConf', err_style='bars',
        linewidth=1, estimator=np.median,
        mew=0.5, marker='o', errorbar=('ci', 68), color='black'
    )
    ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)')
    
    # 3. Plot session
    sns.scatterplot(
        data=subj_data, ax=ax[2],
        x='trials.allN', y='RT_decConf',
        style='Accuracy', hue='Accuracy',
        palette={1.: "#009E73", 0.: "#D55E00"},
        markers={1.: 'o', 0.: 'X'}, s=10, edgecolors='face',
        alpha=0.5, legend=False
    )
    ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")
    
    # Compute running median for filtered data
    running_median = subj_data[['trials.allN', 'RT_decConf']].rolling(10, on='trials.allN').median()
    
    # Plot the running median line on the session subplot
    sns.lineplot(
        data=running_median, ax=ax[2],
        x='trials.allN', y='RT_decConf', color='black', errorbar=None
    )
    
    # Add vertical lines to mark block breaks
    if 'block_breaks' in data.columns:
        subj_block_breaks = subj_data[subj_data['block_breaks'] == 1]
        block_end_trials = subj_block_breaks['trials.allN'].unique()  # Get unique trial numbers
        for x in block_end_trials:
            ax[2].axvline(x=x, color='blue', alpha=0.2, linestyle='--')
    
    # Display the plot
    plt.suptitle(f'Subject {subj} | Accuracy: {accuracy:.2%}')
    plt.show()

# ================================================================================================================

# ================================================================================================================

### Length of experiment 

## Historgam of number of trials 

# Calculate the number of trials per subject 
trials_per_subject = data['trials.allN'].groupby(data['Subj_idx']).count()
trials_per_subject # 700 for all subjects

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(trials_per_subject, bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Trials')
plt.ylabel('Number of Subjects')
plt.title('Histogram of Number of Trials Per Subject')
plt.suptitle('n = 700 for all subjects!')
plt.show()

## Trial length
data['RT_decConf'].mean() # Mean RT
data['RT_decConf'].groupby(data['Subj_idx']).mean() # Mean RT per subject
total_time_per_subject = data.groupby('Subj_idx')['RT_decConf'].sum() #total time spent in the experiment per subject
total_time_per_subject

print("Mean time spent in experiment:", total_time_per_subject.mean())
print("Minimum total time:", total_time_per_subject.min())
print("Maximum total time:", total_time_per_subject.max())
