#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:24:51 2024

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

## Load data
data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Motion discrimination/Hellman_2023/data_Hellmann_2023_Exp2.csv')
data.head()
data.columns # Mutliple sessions!!
data['Stimulus'].unique() # Response is coded as 1 & 2 here, instead of binary 
any(data['RT_decConf'] < 0) # No RT values under 0
max(data['RT_decConf']) # Max RT = 225 > extremely long
data['RT_decConf'].mean() # Mean RT is 2.95 >influenced by extrene values
all(data.notna()) # No NA's in entire dataset 
plt.hist(data['RT_decConf'])


## Pre-processing

# Convert Stimulus and Response into binary variable (for psychometric function)
data['Stimulus'] = data['Stimulus'].replace({1: 0, 2: 1})
data['Response'] = data['Response'].replace({1: 0, 2: 1})

# Exclude training trials
data = data[data['Training'] == 0] 
# Add variable that counts each trial per subject
data['trials.allN'] = data.groupby('Subj_idx').cumcount() + 1 # + 1 bc of Python indexing
# Add 'block_breaks', 80 trials per block
data['block_breaks'] = (data['trials.allN'] % 80 == 0).astype(int)
# Define sensory evidence based on coherence and side
data['sensory_evidence'] = np.where(data['Stimulus'] == 1, data['Coherence'], -data['Coherence']) # Assume that 1 is right here
# Accuracy variable
data['Accuracy'] = (data['Stimulus'] == data['Response']).astype(int) # astype(int), converts true to 1 and false to 0

# Each session consist of 640 trials; 8 blocks * 80 trials (excl. training), some sessions have more trials 
# due to software errors. Some ppts did multiple sessions (up to 3), but vast majority did 1
# think this means we need to select first session for everyone > otherwhise nested structure
# > Account for intra-inidvidual variance
trials_per_subject_session = data.groupby(['Subj_idx', 'Session']).size()

# ================================================================================================================

### Visualize datasets

## Behavioural snapshots 

# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1,1,2], figsize=(20,5))

# 1. psychometric
plot_psychometric(data, ax=ax[0])
ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)',
        ylim=[-0.05, 1.05])

# Chronometric 
sns.lineplot(data = data, ax=ax[1],
             x= 'sensory_evidence', y = 'RT_decConf', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', 
               ylim=[2, 2.8])

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

## Looks good, onky time course doesn't look good because of extreme values. Let's try to remove 
# ================================================================================================================

## Remove outliers
def remove_rt_outliers(df):
    # Calculate median and IQR for RT values per subject
    grouped = df.groupby('Subj_idx')['RT_decConf'].describe()  # Gets the median (50%) and IQR (75% - 25%)
    
    # Rename columns
    grouped = grouped.rename(columns={'50%': 'median', '75%': 'q75', '25%': 'q25'})
    
    # Compute IQR for each subject
    grouped['IQR'] = grouped['q75'] - grouped['q25']
    
    # Merge the IQR and median values back into the original dataframe
    df = df.merge(grouped[['median', 'IQR']], left_on='Subj_idx', right_index=True)
    
    # Step 2: Define outliers as values outside median ± 1.5 * IQR
    lower_bound = df['median'] - 1.5 * df['IQR']
    upper_bound = df['median'] + 1.5 * df['IQR']
    
    # Step 3: Remove rows where RT is outside the bounds for each subject
    df_cleaned = df[(df['RT_decConf'] >= lower_bound) & (df['RT_decConf'] <= upper_bound)]
    
    # Drop the additional columns used for computation
    df_cleaned = df_cleaned.drop(columns=['median', 'IQR'])
    
    return df_cleaned

# Apply the function to remove outliers
data_trimmed = remove_rt_outliers(data)

# Verify by checking the number of rows before and after outlier removal
print(f"Original data size: {len(data)}")
print(f"Cleaned data size: {len(data_trimmed)}") 
40705 - 35799 # 4906 trials removed
(4906 / 40705)*100 # 12.05 % removed
plt.hist(data_trimmed['RT_decConf'])
data_trimmed['RT_decConf'].mean() # 2.41, removal of extreme scores doesn't affect mean RT to a large extent

## Visualizations

# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1,1,2], figsize=(20,5))

# 1. psychometric
plot_psychometric(data_trimmed, ax=ax[0])
ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)',
        ylim=[-0.05, 1.05])

# Chronometric 
sns.lineplot(data = data_trimmed, ax=ax[1],
             x= 'sensory_evidence', y = 'RT_decConf', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', 
               ylim=[2, 2.5])

# Plot session
sns.scatterplot(data = data_trimmed[data_trimmed['Subj_idx'] == 1], ax=ax[2],
                x='trials.allN', y='RT_decConf',                  
                 style='Accuracy', hue='Accuracy',
                 palette={1.:"#009E73", 0.:"#D55E00"}, 
                 markers={1.:'o', 0.:'X'}, s=10, edgecolors='face',
                 alpha=.5, legend=False)
ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

# Compute running median for filtered data
subject_data = data_trimmed[data_trimmed['Subj_idx'] == 1][['trials.allN', 'RT_decConf']]
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

## Behavioural snapshots - per participant

# Filter dataset to include only Session 1
data_session1 = data_trimmed[data_trimmed['Session'] == 1]

# Randomly select 15 subjects
random_subjects = np.random.choice(data_session1['Subj_idx'].unique(), size=15, replace=False)

# Loop over the randomly selected subjects
for subj in random_subjects:
    # Filter data for the current subject and session 1
    subj_data = data_session1[data_session1['Subj_idx'] == subj]

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
    plt.suptitle(f'Subject {subj}')
    plt.show()
    
# ==============================================================================================================

### Length of experiment 

# Use untrimmed data for this! RT trimming was only for session plot
data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Motion discrimination/Hellman_2023/data_Hellmann_2023_Exp2.csv')
data = data[data['Training'] == 0] # Exclude training data
data['trials.allN'] = data.groupby('Subj_idx').cumcount() + 1 # Count trials again
data_session1 = data[data['Session'] == 1] # Use only session 1 (See above)

## Historgam of number of trials 

# Calculate the number of trials per subject
trials_per_subject = data_session1['trials.allN'].groupby(data_session1['Subj_idx']).count()

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(trials_per_subject, bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Trials')
plt.ylabel('Number of Subjects')
plt.title('Histogram of Number of Trials Per Subject')
plt.suptitle('n = 640 for 38 subjects, 4 deviate.')
plt.show()

## Blcoks per subject -  This is not correct but I know there's 8 blocks anyway 
data['block_breaks'] = ((data.index + 1) % 80 == 0).astype(int) 
blocks_per_subject = data[data['block_breaks'] == 1].groupby(data['Subj_idx']).count()
all(blocks_per_subject == 6)

## Trial length
data_session1['RT_decConf'].mean() # Mean RT
data_session1['RT_decConf'].groupby(data_session1['Subj_idx']).mean() # Mean RT per subject
total_time_per_subject = data_session1.groupby('Subj_idx')['RT_decConf'].sum() #total time spent in the experiment per subject
total_time_per_subject

print("Mean time spent in experiment:", total_time_per_subject.mean())
print("Minimum total time:", total_time_per_subject.min())
print("Maximum total time:", total_time_per_subject.max())

# ==============================================================================================================
## Data artefacts 


# ================================================================================================================

