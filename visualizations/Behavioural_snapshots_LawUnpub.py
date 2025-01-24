#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:39:44 2024

@author: lorenzovanhoorde

Behavioural visualizations for model fitting
"""
### Import packages and define functiomns

import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl# Set up figure and axes

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

# =====================================signed_contrast========================================================================

## Load data
data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Motion discrimination/Law_unpub/data_Law_unpub.csv')
data.head()
data.columns
data['Stimulus'].unique() # Response is coded as 1 & 2 here, instead of binary 
any(data['RT_decConf'] < 0) # No RT values under 0
max(data['RT_decConf']) # Max RT = 30.7 > extremely long, 12.3 after removal of calibration trials
data['RT_decConf'].mean() # Mean RT is 0.75
all(data.notna()) # No NA's in entire dataset
plt.hist(data['RT_decConf']) 

## Pre-processing

# Remove calibration trials
data = data[data['isCalibTrial'] == 0]
# Convert Stimulus and Response into binary variable (for psychometric function)
data['Stimulus'] = data['Stimulus'].replace({1: 0, 2: 1})
data['Response'] = data['Response'].replace({1: 0, 2: 1})

# Indicate breaks between blocks (60 trials per block)
data['block_breaks'] = ((data.index + 1) % 60 == 0).astype(int) 
# Define signed contrast based on coherence and side
data['sensory_evidence'] = np.where(data['Stimulus'] == 1, data['coherence'], -data['coherence'])
data['signed_coh_level'] = np.where(data['Stimulus'] == 1, data['coh_level'], -data['coh_level']) # Used later on
# Add variable that counts each trial per subject
data['trials.allN'] = data.groupby('Subj_idx').cumcount() + 1 # + 1 bc of Python indexing
# Accuracy variable
data['Accuracy'] = (data['Stimulus'] == data['Response']).astype(int) # astype(int), converts true to 1 and false to 0

# ================================================================================================================

### Visualize datasets

## Behavioural snapshots 

# Set up figure and axes
fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1,1,2], figsize=(20,5))

# 1. psychometric
plot_psychometric(data, ax=ax[0])
ax[0].set(xlabel='Signed contrast', ylabel='Choice (fraction)',
        ylim=[-0.05, 1.05])

# Chronometric 
sns.lineplot(data = data, ax=ax[1],
             x= 'sensory_evidence', y = 'RT_decConf', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Signed contrast', ylabel='Response time (s)', 
               ylim=[0, 1.8])

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

## Looks okay, not the best but might be usable 

# ===============================================================================================================

# ===============================================================================================================

### Step 1: Remove outliers for session plot

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
15360 - 14018 # 1342 trials removed
(1342/15360) * 100 # 8.74% trials removed
plt.hist(data_trimmed['RT_decConf'])
data_trimmed['RT_decConf'].mean() # 0.56, removal of extreme scores doesn't affect mean RT to a large extent

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
               ylim=[0, 1.8])

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

# TODO: Add blocks > think they have this in the data 
# add lines to mark breaks, if these were included
if 'block_breaks' in data.columns:
    if data['block_breaks'].iloc[0] == 'y':
        [ax[1].axvline(x, color='blue', alpha=0.2, linestyle='--') 
             for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
plt.show()

# ===============================================================================================================

# ===============================================================================================================

### Step 2: Try to bin signed contrast

# Define the bin edges to cover the full range from -1 to 1 (with the correct intervals)
bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Define the corresponding labels based on coherence levels
labels = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

data_trimmed['coherence_binned'] = pd.cut(data_trimmed['coherence'], bins=bins, labels=labels, right=False)
data_trimmed['coherence_binned'] = data_trimmed['coherence_binned'].astype(float)
data_trimmed['sensory_evidence'] = np.where(data_trimmed['Stimulus'] == 1, data_trimmed['coherence_binned'], -data_trimmed['coherence_binned'])

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
               ylim=[0.3, 0.7])

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

# TODO: Add blocks > think they have this in the data 
# add lines to mark breaks, if these were included
if 'block_breaks' in data.columns:
    if data['block_breaks'].iloc[0] == 'y':
        [ax[1].axvline(x, color='blue', alpha=0.2, linestyle='--') 
             for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
plt.show()

# ===============================================================================================================

# ===============================================================================================================

### Extra:
    
# Chronometric, curve with coh_levels >>> see plots right
sns.lineplot(data = data,
             x= 'signed_coh_level', y = 'RT_decConf', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', 
               ylim=[0, 1.8]) 
plt.show()

## On trimmed data
sns.lineplot(data = data_trimmed,
             x= 'signed_coh_level', y = 'RT_decConf', err_style = 'bars',
             linewidth=1, estimator = np.median,
             mew=0.5, marker ='o', errorbar=('ci',68), color='black')
ax[1].set(xlabel='Sensory evidence', ylabel='Response time (s)', 
               ylim=[0, 1.8]) 
plt.show()


data['sensory_evidence'] = data['signed_coh_level']
data_trimmed['sensory_evidence'] = data_trimmed['signed_coh_level']
# 1. psychometric
plot_psychometric(data) # Actually doesn't look that bad 
plot_psychometric(data_trimmed)

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
               ylim=[0.42, 0.56])

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



### If you run plotting code now, you see plot right. Very much not steep. Doesn't really look good. 

# ===============================================================================================================
# ================================================================================================================

## Behavioural snapshots - per participant

random_subjects = np.random.choice(data_trimmed['Subj_idx'].unique(), size=15, replace=False)

# Loop over the randomly selected subjects
for subj in random_subjects:
    # Filter data for the current subject
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
    
    
# Inspoect accuracy > added to the plot
data['Accuracy'].groupby(data['Subj_idx']).mean() # Mean ACC per subject, untrimmed data > acc slightly lower
## People with worse psychometric curves perform badly, people with good curves perform well
# ===============================================================================================================

# ==============================================================================================================

### Length of experiment 

## Historgam of number of trials 

# Calculate the number of trials per subject - Perform on UNTRIMMED DATA
trials_per_subject = data['trials.allN'].groupby(data['Subj_idx']).count()
trials_per_subject

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(trials_per_subject, bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Trials')
plt.ylabel('Number of Subjects')
plt.title('Histogram of Number of Trials Per Subject')
plt.suptitle('n = 960 for all subjects!')
plt.show()

## Blcoks per subject
blocks_per_subject = data[data['block_breaks'] == 1]['block_breaks'].groupby(data['Subj_idx']).count()
blocks_per_subject
all(blocks_per_subject == 16) # All subjects have 16 blocks 

## Trial length
data['RT_decConf'].mean() # Mean RT
data['RT_decConf'].groupby(data['Subj_idx']).mean() # Mean RT per subject
total_time_per_subject = data.groupby('Subj_idx')['RT_decConf'].sum() #total time spent in the experiment per subject
total_time_per_subject

print("Mean time spent in experiment:", total_time_per_subject.mean())
print("Minimum total time:", total_time_per_subject.min())
print("Maximum total time:", total_time_per_subject.max())

# ==============================================================================================================

## Don't really use this anymore 


# Function to compute signed contrast
def compute_signed_contrast(df):
    # Step 1: Map coherence levels to signed contrast
    # Ensure coherence corresponds to its respective accTarget
    coherence_map = (
        df[df['isCalibTrial'] == 1]  # Use only calibration trials to establish mapping
        .groupby('accTarget')['coherence']
        .mean()  # Average coherence for each accuracy target
    )
    
    # Define signed contrast variable
    df['signed_contrast'] = np.where(
        df['Stimulus'] == 0,  # Leftward stimulus
        -df['coherence'],  # Negative for left
        df['coherence']  # Positive for right
    )

    # Step 2: Normalize signed contrast to standardized levels
    # Create a dictionary to map coherence to signed contrast levels
    normalized_levels = np.linspace(-2, 2, len(coherence_map))
    coherence_to_signed_contrast = dict(zip(coherence_map.sort_values(), normalized_levels))

    # Replace signed contrast with normalized levels
    df['signed_contrast'] = df['coherence'].map(coherence_to_signed_contrast)
    df['signed_contrast'] = np.where(
        df['Stimulus'] == 0,  # Ensure leftward is negative
        -df['signed_contrast'],
        df['signed_contrast']
    )

    return df, coherence_map, coherence_to_signed_contrast

# Define signed contrast based on coherence and side
data['signed_contrast'] = data['coh_level'].replace({1:-0.2, 2:-0.1, 3: 0, 4: 0.1, 5: 0.2})
# The coherence levels map to different mean ACC levels per subject, that were calibrated before real blcoks
# So this reflects trial difficulty
# Can use other ways to calculate signed contrast later, but this should be fine for plotting



