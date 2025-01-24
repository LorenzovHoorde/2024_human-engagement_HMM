#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:21:59 2025

@author: lorenzovanhoorde
"""
# ============================================================================================================

### Load data and packages
# ============================================================================================================

## Packages
from __future__ import  division

import json
import warnings


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm



from patsy import dmatrices
import random

from IOHMM import UnSupervisedIOHMM
from IOHMM import GLM, OLS, DiscreteMNL, CrossEntropyMNL
import psychofit as psy 

warnings.simplefilter("ignore")

# ============================================================================================================

## Define functions

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
    

# ============================================================================================================

## Load data
data = pd.read_csv('/home/lorenzovanhoorde/Documenten/Thesis_Human-engagement_HMM/Datasets/Final_sample/test_sample/data_Zylberberg_2016.csv')
data.columns

# ============================================================================================================

## Filter data

# Select 1st date for every ppt
data['Date'].unique() # 7 separate testing days
data['Date'].groupby(data['Subj_idx']).unique() # 4 days for every ppt, 1st day not the same for all ppts
first_dates = data.groupby('Subj_idx')['Date'].min()
data = data[data['Date'] == data['Subj_idx'].map(first_dates)] # Adjust DF

## Data checks

data['Stimulus'].unique() # Should be (and is) binary
any(data['RT_decConf'] < 0) # No RT values under 0
max(data['RT_decConf']) # Max RT = 4.02
data['RT_decConf'].mean() # Mean RT is 0.95
any(data['RT_decConf'].isna()) # Responses NaN if subjects failed to maintain fixation during the motion epoch.
data.isna().sum() # 2 NA's > no response recorded (and hence no RT as well), we can remove these
data = data.dropna()

## Pre-processing

# Add variable that counts each trial per subject
data['trial'] = data.groupby('Subj_idx').cumcount() + 1 # + 1 bc of Python indexing
# Compute block breaks, might be useful for later
data['block_breaks'] = data.groupby('Subj_idx')['Block'].diff().ne(0).astype(int)
# Add sensory evidence variable (= signed coherence) > GLM covariate
data['sensory_evidence'] = np.where(data['Stimulus'] == 1, data['Coherence'], -data['Coherence'])
# Add previous choice variable > GLM covariate
data['previous_choice'] = data.groupby('Subj_idx')['Response'].shift(1)
# Gives 3 extra NA's tho, remove! Pretty common to remove first trial
data.isna().sum()

# Drop unnecessary variables + rename RT
data = data.drop(columns = ["Confidence", "Date", "Coherence"])
data.rename(columns={"RT_decConf": "RT"}, inplace=True)

df = data.dropna() # Drops first trial for every subject

# ============================================================================================================

### Fit 1-state GLM - Fit this as regular logistic regression, since 1-state GLM-HMM = just GLM

## General fit

# Define inputs and response
y,x = dmatrices('Response ~ sensory_evidence + previous_choice + Condition', # dmatrices ensures intercept
                data = df, return_type = 'dataframe')

# fit log. regression model
random.seed(42)
log_reg_allN = sm.Logit(y, x).fit()
print(log_reg_allN.summary())


## Fit per subject separately

subjects = df['Subj_idx'].unique()
log_reg = {}

for subj in subjects:
    
    # Define inputs and response
    subj_data = df[df['Subj_idx'] == subj]
    y,x = dmatrices('Response ~ sensory_evidence + previous_choice + Condition', # dmatrices ensures intercept
                    data = subj_data, return_type = 'dataframe')
    
    # Fit GLM (logistic regression)
    random.seed(42)
    log_reg[subj] = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    print(log_reg[subj].summary())
    
    # Prepare for plotting
    coefficients = log_reg[subj].params
    covariates = ['Bias', 'Stim.', 'p.c.', 'Cond']
    coefficients_df = pd.DataFrame({'Covariate': covariates, 'Coefficient': coefficients.values})
    
    # Plots
    fig, ax = plt.subplots(ncols=2, nrows=1, width_ratios=[1, 1], figsize=(20, 5)) # Set up figure and axes
    
    # Psychometric curve
    plot_psychometric(subj_data, ax=ax[0])
    ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', ylim=[-0.05, 1.05])
    
    # Plot GLM coefficients
    sns.lineplot(data=coefficients_df, x='Covariate', y='Coefficient', ax=ax[1], marker='o')
    ax[1].set(xlabel='Covariates', ylabel='GLM Coefficients', title=f'Subject {subj}')
    
## Construct plot across subjects 

coef = log_reg_allN.params
covariates = ['Bias', 'Stim.', 'p.c.', 'Cond']
coef_df = pd.DataFrame({'Covariate': covariates, 'Coefficient': coef.values})

fig, ax = plt.subplots(ncols=2, nrows=1, width_ratios=[1, 1], figsize=(20, 5)) # Set up figure and axes

plot_psychometric(df, ax=ax[0])
ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', ylim=[-0.05, 1.05])

sns.lineplot(data=coef_df, x='Covariate', y='Coefficient', ax=ax[1], marker='o')
ax[1].set(xlabel='Covariates', ylabel='GLM Coefficients')

plt.show()

# ============================================================================================================

### 1 state GLM using IOHMM package

SHMM_1 = UnSupervisedIOHMM(num_states=1, max_EM_iter=200, EM_tol=1e-6) # 1 state

# This model has only one output which is modeled by a logistic regression model
SHMM_1.set_models(model_emissions = [GLM(family=sm.genmod.families.Binomial())], # logistic regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_1.set_inputs(covariates_initial = [], covariates_transition = [], 
                  covariates_emissions = [['sensory_evidence', 'previous_choice', 'Condition']])  
SHMM_1.set_outputs([['Response']]) # we only have a list of one sequence.

SHMM_1.set_data([df]) # Specify dataframe
random.seed(42)
SHMM_1.train() # Run model

print(SHMM_1.model_emissions[0][0].coef) # Same coefficients as log regression fit, as it should

# ============================================================================================================

### 2 state GLM-HMM

# ============================================================================================================

### General fit

SHMM_2 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states

# This model has only one output which is modeled by a linear regression model
SHMM_2.set_models(model_emissions = [GLM(family=sm.genmod.families.Binomial())], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_2.set_inputs(covariates_initial = [], covariates_transition = [], 
                  covariates_emissions = [['sensory_evidence', 'previous_choice', 'Condition']])  
SHMM_2.set_outputs([['Response']]) # we only have a list of one sequence.

SHMM_2.set_data([df]) # Specify dataframe
random.seed(42)
SHMM_2.train() # Run model

# State-specific GLM
print(SHMM_2.model_emissions[0][0].coef) # More weight on the stimulus + larger weight previous choice
print(SHMM_2.model_emissions[1][0].coef) # Bias towards other side (left), other weights smaller

# Transition probabilities
print(np.exp(SHMM_2.model_transition[0].predict_log_proba(np.array([[]])))) #50-50
print(np.exp(SHMM_2.model_transition[1].predict_log_proba(np.array([[]])))) # Close to 50-50

# 2-state solution seems to yield a 'choose right' (state 1) and 'choose left' (state 2) state
# With a 50-50 transition matrix, which makes sense since left and right stimuli should be as frequent 

## Prepare for plotting

# Since we want to make plots for each state separately, we need to know which trials belong to which state
# Luckily we can calculate this

# Extract state predictions

predicted_states = []

for sequence_idx, log_gamma in enumerate(SHMM_2.log_gammas):
    # Exponentiate log_gamma to get probabilities
    gamma_probs = np.exp(log_gamma)
    # Find the most likely state for each timestep in the sequence
    sequence_states = np.argmax(gamma_probs, axis=1)
    # Extend predicted states for all sequences
    predicted_states.extend(sequence_states)

# Add to the original dataframe and construct separate dataframes
df['State_pred'] = predicted_states
df_s1 = df[df['State_pred'] == 0]
df_s2 = df[df['State_pred'] == 1]

# Get GLM weights
covariates = ['Bias', 'Stim.', 'p.c.', 'Cond']
coef_s1 = pd.DataFrame({'Covariate': covariates, 'Coefficient': SHMM_2.model_emissions[0][0].coef})
coef_s2 = pd.DataFrame({'Covariate': covariates, 'Coefficient': SHMM_2.model_emissions[1][0].coef})

## Plot

fig, ax = plt.subplots(ncols=2, nrows=2, width_ratios=[1, 1], figsize=(15, 15)) # Set up figure and axes

plot_psychometric(df_s1, ax=ax[0, 0])
ax[0, 0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', title='Psych curve state 1',
          ylim=[-0.05, 1.05])

sns.lineplot(data=coef_s1, x='Covariate', y='Coefficient', ax=ax[0, 1], marker='o', color='blue')
ax[0, 1].set(xlabel='Covariates', ylabel='GLM Coefficients', title='GLM weights state 1')

plot_psychometric(df_s2, ax=ax[1, 0])
ax[1, 0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', title='Psych curve state 2',
          ylim=[-0.05, 1.05])

sns.lineplot(data=coef_s2, x='Covariate', y='Coefficient', ax=ax[1, 1], marker='o', color='orange')
ax[1, 1].set(xlabel='Covariates', ylabel='GLM Coefficients', title='GLM weights state 2')

plt.show()

# ============================================================================================================

### Fit per subject separately

subjects = df['Subj_idx'].unique()
s2_reg = {}

for subj in subjects:
    
    # fit model on subject data
    subj_data = df[df['Subj_idx'] == subj]
    SHMM_2.set_data([subj_data])
    random.seed(42)
    SHMM_2.train()
    s2_reg[subj] = SHMM_2
    
    # Print indvidual coefficients
    print(f'Subject {subj}')
    print("State 1", s2_reg[subj].model_emissions[0][0].coef)
    print("State 2", s2_reg[subj].model_emissions[1][0].coef)
    
    # Extract predicted states
    state_pred = []
    state_prob = []
    for _, log_gamma in enumerate(SHMM_2.log_gammas):  # Unpack index and log_gamma
        prob = np.exp(log_gamma)
        state_seq = np.argmax(prob, axis=1)  # Find most likely state
        state_pred.extend(state_seq)  # Add to predictions
        state_prob.extend(prob)
        
    # Convert state_probs into a 2D array
    state_probs = np.vstack(state_prob)  # Shape: (n_trials, n_states)
    
    # Add to subject data, and split dataframe
    subj_data['State_prob_0'] = state_probs[:, 0]  # Probability for State 0
    subj_data['State_prob_1'] = state_probs[:, 1]  # Probability for State 1
    subj_data['State_pred'] = state_pred
    subj_s1 = subj_data[subj_data['State_pred'] == 0]
    subj_s2 = subj_data[subj_data['State_pred'] == 1]    
    
    # Get GLM weights
    covariates = ['Bias', 'Stim.', 'p.c.', 'Cond']
    coef_s1 = pd.DataFrame({'Covariate': covariates, 'Coefficient': s2_reg[subj].model_emissions[0][0].coef})
    coef_s2 = pd.DataFrame({'Covariate': covariates, 'Coefficient': s2_reg[subj].model_emissions[1][0].coef})
    
    # Combine coefficients into a single DataFrame
    coef_s1['State'] = 'State 1'
    coef_s2['State'] = 'State 2'
    coef_combined = pd.concat([coef_s1, coef_s2], ignore_index=True)
    
    # Plot 1: Psychometric curve and GLM weights per state
    fig, ax = plt.subplots(ncols=3, nrows=1, width_ratios=[1, 1, 2], figsize=(20, 8)) # Set up figure and axes

    plot_psychometric(subj_s1, ax=ax[0])
    ax[0].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', title='Psych curve state 1',
              ylim=[-0.05, 1.05])

    plot_psychometric(subj_s2, ax=ax[1])
    ax[1].set(xlabel='Sensory evidence', ylabel='Choice (fraction)', title='Psych curve state 2',
              ylim=[-0.05, 1.05])

    sns.lineplot(data=coef_combined,x='Covariate',y='Coefficient', hue='State',
        palette=['blue', 'orange'], ax=ax[2], marker='o')
    ax[2].set(xlabel='Covariates',ylabel='GLM Coefficients',title='GLM weights for both states')
    
    # Add a suptitle for the figure
    fig.suptitle(f'Subject {subj}', fontsize=16)

    plt.show()
    
    # Plot 2: Posterior state probabilities + raw behavioural data
    fig, ax = plt.subplots(ncols=2, nrows=1, width_ratios=[1, 1], figsize=(20, 8)) # Set up figure and axes

    sns.lineplot(data=subj_data.melt(id_vars=['trial'], value_vars=['State_prob_0', 'State_prob_1'], 
                var_name='State', value_name='Probability'),x='trial', y='Probability', hue='State',
    palette=['blue', 'orange'], ax=ax[0], marker=False)
    ax[0].set(xlabel='Trial #',ylabel='p(state)',title='Posterior state probablities')
    
    sns.scatterplot(data=subj_data,x='trial',y='Response',hue='Accuracy',
        palette=['red', 'black'],ax=ax[1],marker='o')
    ax[1].set(xlabel='Trial #',ylabel='Choice',title='Behavioural data')
        
    # Add a suptitle for the figure
    fig.suptitle(f'Subject {subj}', fontsize=16)
    
    plt.show()