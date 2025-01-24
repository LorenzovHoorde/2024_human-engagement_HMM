#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:18:15 2024

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

## Load data and pre-processing

data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Orientation discrimination/Rausch_2016/data_Rausch_2016.csv')
data.head()
data.columns
data['contrast'].unique() # Few discrete levels > good!

# Exclude training data
data = data[data['Training'] == 0]

# Inpsect coding and RT
data['Stimulus'].unique() # Coded as 1 and 2 here 
any(data['RT_dec'] < 0) # No RT values under 0
max(data['RT_dec']) # Max RT = 17.7 > quite long
data['RT_dec'].sort_values(ascending=False) # Quite okay for being the largest, also no implausibly quick RTS
data['RT_dec'].mean() # Mean RT is 1.05
any(data['RT_dec'].isna()) # No NA's


## Pre-processing

# Convert Stimulus and Response into binary variable (for psychometric function)
data['Stimulus'] = data['Stimulus'].replace({1: 0, 2: 1})
data['Response'] = data['Response'].replace({1: 0, 2: 1})

# Define sensory evidence based on contrast and side
data['sensory_evidence'] = np.where(data['Stimulus'] == 1, data['contrast'], -data['contrast'])
# Add variable that counts each trial per subject
data['trials.allN'] = data.groupby('Subj_idx').cumcount() + 1 # + 1 bc of Python indexing
# Indicate breaks between blocks (42 trials per block)
data['block_breaks'] = ((data.index + 1) % 42 == 0).astype(int)

max(data['trials.allN']) # 378, pushes the limit a little bit 

# =============================================================================================================

### Visualize datasets

## Behavioural snapshots - average

# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1,1,2], figsize=(20,5))

# 1. psychometric
plot_psychometric(data, ax=ax[0])
ax[0].set(xlabel='Sesnory evidence', ylabel='Choice (fraction)',
        ylim=[-0.05, 1.05])

# Chronometric 
sns.lineplot(data = data, ax=ax[1],
             x= 'sensory_evidence', y = 'RT_dec', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', 
               ylim=[0.4, 1.2])

# Plot session
sns.scatterplot(data = data[data['Subj_idx'] == 1], ax=ax[2],
                x='trials.allN', y='RT_dec',                  
                 style='Accuracy', hue='Accuracy',
                 palette={1.:"#009E73", 0.:"#D55E00"}, 
                 markers={1.:'o', 0.:'X'}, s=10, edgecolors='face',
                 alpha=.5, legend=False)
ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

# Compute running median for filtered data
subject_data = data[data['Subj_idx'] == 1][['trials.allN', 'RT_dec']]
running_median = subject_data.rolling(10, on='trials.allN').median()

# Plot the running median line on the second subplot
sns.lineplot(
    data=running_median, ax=ax[2],
    x='trials.allN', y='RT_dec', color='black', errorbar=None
)

# Add vertical lines to mark block breaks
# Filter block breaks for the specific subject
if 'block_breaks' in data.columns:
    subj_block_breaks = data[(data['block_breaks'] == 1) & (data['Subj_idx'] == 1)]
    block_end_trials = subj_block_breaks['trials.allN'].unique()  # Get unique trial numbers
    for x in block_end_trials:
        ax[2].axvline(x=x, color='blue', alpha=0.2, linestyle='--')


plt.show() 

# =============================================================================================================

# Trim data just for plots > quite some extreme RTs 

## Remove outliers
def remove_rt_outliers(df):
    # Calculate median and IQR for RT values per subject
    grouped = df.groupby('Subj_idx')['RT_dec'].describe()  # Gets the median (50%) and IQR (75% - 25%)
    
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
    df_cleaned = df[(df['RT_dec'] >= lower_bound) & (df['RT_dec'] <= upper_bound)]
    
    # Drop the additional columns used for computation
    df_cleaned = df_cleaned.drop(columns=['median', 'IQR'])
    
    return df_cleaned

# Apply the function to remove outliers
data_trimmed = remove_rt_outliers(data)

# Verify by checking the number of rows before and after outlier removal
print(f"Original data size: {len(data)}")
print(f"Cleaned data size: {len(data_trimmed)}") 
7560 - 6747 # 813 trials removed
(813 / 7560)*100 # 10.75 % removed
plt.hist(data_trimmed['RT_dec'])
data_trimmed['RT_dec'].mean() # 0.95

# ================================================================================================================

## Visualizations

# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1,1,2], figsize=(20,5))

# 1. psychometric
plot_psychometric(data_trimmed, ax=ax[0])
ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)',
        ylim=[-0.05, 1.05])

# Chronometric 
sns.lineplot(data = data_trimmed, ax=ax[1],
             x= 'sensory_evidence', y = 'RT_dec', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', 
               ylim=[0.7, 1.0])

# Plot session
sns.scatterplot(data = data_trimmed[data_trimmed['Subj_idx'] == 1], ax=ax[2],
                x='trials.allN', y='RT_dec',                  
                 style='Accuracy', hue='Accuracy',
                 palette={1.:"#009E73", 0.:"#D55E00"}, 
                 markers={1.:'o', 0.:'X'}, s=10, edgecolors='face',
                 alpha=.5, legend=False)
ax[2].set(xlabel="Trial number", ylabel="Reaction times (s)")

# Compute running median for filtered data
subject_data = data_trimmed[data_trimmed['Subj_idx'] == 1][['trials.allN', 'RT_dec']]
running_median = subject_data.rolling(10, on='trials.allN').median()

# Plot the running median line on the second subplot
sns.lineplot(
    data=running_median, ax=ax[2],
    x='trials.allN', y='RT_dec', color='black', errorbar=None
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

# Select 15 random subjects
random_subjects = np.random.choice(data_trimmed['Subj_idx'].unique(), size=15, replace=False)

# Loop over the randomly selected subjects
for subj in random_subjects:
    # Filter data for the current subject and session 1
    subj_data = data_trimmed[data_trimmed['Subj_idx'] == subj]
    
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

    # Display the plot
    plt.suptitle(f'Subject {subj} | Accuracy: {accuracy:.2%}')
    plt.show()
    
    
# ==============================================================================================================

### Length of experiment, perform on UNTRIMMED data

## Historgam of number of trials 

# Calculate the number of trials per subject
trials_per_subject = data['trials.allN'].groupby(data['Subj_idx']).count()
trials_per_subject

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(trials_per_subject, bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Trials')
plt.ylabel('Number of Subjects')
plt.title('Histogram of Number of Trials Per Subject')
plt.suptitle('n = 378 for all subjects!')
plt.show()

## Blocks per subject
max(data['Block']) # 12 blocks

## Trial length
data['RT_dec'].mean() # Mean RT
data['RT_dec'].groupby(data['Subj_idx']).mean() # Mean RT per subject
total_time_per_subject = data.groupby('Subj_idx')['RT_dec'].sum() #total time spent in the experiment per subject
total_time_per_subject

print("Mean time spent in experiment:", total_time_per_subject.mean())
print("Minimum total time:", total_time_per_subject.min())
print("Maximum total time:", total_time_per_subject.max())
