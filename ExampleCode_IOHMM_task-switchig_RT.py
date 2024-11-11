#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:00:15 2024

@author: lorenzovanhoorde

Fit IOHMM to task-switching data 
- RT analysis

"""
### Comments ================================================================================================

# Anoying thing is that despite setting a random seed before fitting the models, the order of states (i.e. 
# which state is number 1 and 2) differs between runs 

# Load data and packages ====================================================================================

### Main interpeter should be codingcody/bin/python

from __future__ import  division

import json
import warnings


import numpy as np
import pandas as pd
import random


from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, DiscreteMNL, CrossEntropyMNL
warnings.simplefilter("ignore")

def model_attributes(SHMM, dataframe):
    """
    Calculate model attributes for model comparison.
    """
    
    ## Callculate number of parameters ========================================
    # Number of hidden states and covariates
    num_states = len(SHMM.model_transition)
    num_transition_covariates = len(SHMM.covariates_transition)
    num_emission_covariates = len(SHMM.covariates_emissions[0])

    # Transition parameters: each state transition except self-transitions
    transition_params = num_states * (num_states - 1) * (num_transition_covariates + 1)

    # Emission parameters: adapt as necessary based on model attribute structure
    emission_params = num_states * (num_emission_covariates + 1)
    
    # Initial state distribution parameters: K - 1 for K states
    initial_state_params = num_states - 1

    # Total parameters
    num_params = transition_params + emission_params + initial_state_params
    # =========================================================================
    # Observations
    num_observations = len(dataframe)

    # Log-likelihood from SHMM
    log_likelihood = getattr(SHMM, 'log_likelihood', None)

    # Return results as tuple
    return log_likelihood, num_params, num_observations


def calculate_aic_bic(log_likelihood, num_params, num_observations):
    """
    Calculate the AIC and BIC for a model.

    Parameters:
    - log_likelihood: float, the log-likelihood of the model
    - num_params: int, the number of estimated parameters in the model
    - num_observations: int, the number of observations (data points)

    Returns:
    - aic: float, the Akaike Information Criterion
    - bic: float, the Bayesian Information Criterion
    """
    # AIC calculation
    aic = 2 * num_params - 2 * log_likelihood

    # BIC calculation
    bic = num_params * np.log(num_observations) - 2 * log_likelihood

    return aic, bic

### Load data

mydata = pd.read_csv('/home/lorenzovanhoorde/Documenten/Master_Cognitive_Neuorscience/Stage/Data/Raw data/dataTaskSwitching_firstBlockExcluded.csv')
mydata.head

# Some pre-processing 

mydata[mydata['Switch'].isna() == True] # Trial 1 is NA in every block, because no switch information
any(mydata.drop(columns=['Switch']).isna()) # Also NA's in other variables tho

# What are the other NA's??
rows_with_na_outside_switch = mydata.drop(columns=['Switch']).isna().any(axis=1)
mydata[rows_with_na_outside_switch] # 43 rows, all missing RT's and ACC = 0, maybe trials without timely response?

# Convert variables to numerical
mydata['Switch'].replace(['task-switch', 'task-repetition'], [1, 0], inplace=True) # Convert switch to binary var
mydata['Reward'].replace(['Reward', 'No-reward'], [1, 0], inplace=True) # Convert switch to binary var.
mydata['Congruence'].replace(['congruent', 'incongruent'], [1, 0], inplace=True)
mydata['Sequence'].replace(['Fake Frequent', 'Fake Rare', 'No Instruction', 'Real Frequent', 'Real Rare'],
                           [0, 1, 2, 3, 4], inplace=True) # Convert sequence to categorical var.
# I used this order because it is the order in which the sequences occur throughout the experiment
# TODO: Think about whether this order is the most sensible one 

# Drop NA's print("Best model based on log-likelihood is Model", best_model_index + 1)
df = mydata.dropna()

# Testing version control Git!!

# ===================================================================================================================

# Model 1: Only RT, no covariates
SHMM_1 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states

# This model has only one output which is modeled by a linear regression model
SHMM_1.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_1.set_inputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])  # No covariates
SHMM_1.set_outputs([['RT']]) # we only have a list of one sequence.

random.seed(42)
SHMM_1.set_data([df]) # Specify dataframe
SHMM_1.train() # Run model

print(SHMM_1.model_emissions[0][0].coef) # Intercept for state 1, 1210 RT > slow responding
print(SHMM_1.model_emissions[1][0].coef) # Intecept for state 2, 676 RT > fast responding

# The scale/dispersion of the OLS model for each hidden states
print(np.sqrt(SHMM_1.model_emissions[0][0].dispersion))
print(np.sqrt(SHMM_1.model_emissions[1][0].dispersion))

# The transition probability between two hidden states
print(np.exp(SHMM_1.model_transition[0].predict_log_proba(np.array([[]])))) # 80.9% chance of staying in slow resp.
print(np.exp(SHMM_1.model_transition[1].predict_log_proba(np.array([[]])))) # 93.4% chance of staying in fast resp.

#  =================================================================================================================

### Model 2 - Only RT, switch as emission covariate
SHMM_2 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states

# This model has only one output which is modeled by a linear regression model
SHMM_2.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_2.set_inputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [['Switch'], []]) 
SHMM_2.set_outputs([['RT']]) # we only have a list of one sequence.

SHMM_2.set_data([df]) # Specify dataframe
random.seed(42)
SHMM_2.train() # Run model

# GLM 
print(SHMM_2.model_emissions[0][0].coef) # Intercept for state 1, 636 RT, 86 ms switch-cost > fast responding
print(SHMM_2.model_emissions[1][0].coef) # Intecept for state 2, 1151 RT, 125 ms switch-cost > slow responding

print(np.sqrt(SHMM_2.model_emissions[0][0].dispersion))
print(np.sqrt(SHMM_2.model_emissions[1][0].dispersion))

# The transition probability between two hidden states
print(np.exp(SHMM_2.model_transition[0].predict_log_proba(np.array([[]])))) # 93.2% chance of stayin in fast resp.
print(np.exp(SHMM_2.model_transition[1].predict_log_proba(np.array([[]])))) # 79.8% chance of staying in slow resp.

# But what drives these switches in latent strategies?? 

# ===============================================================================================================

### Model 3 - Only RT, siwtch as transition covariate

SHMM_3 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states

# This model has only one output which is modeled by a linear regression model
SHMM_3.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_3.set_inputs(covariates_initial = [], covariates_transition = ['Switch'], covariates_emissions = [[]])
SHMM_3.set_outputs([['RT']])

random.seed(42)
SHMM_3.set_data([df])
SHMM_3.train()

# GLM 
print(SHMM_3.model_emissions[0][0].coef) # Intercept for state 1, 672 RT fast responding
print(SHMM_3.model_emissions[1][0].coef) # Intecept for state 2, 1206 RT slow responding

print(np.sqrt(SHMM_3.model_emissions[0][0].dispersion))
print(np.sqrt(SHMM_3.model_emissions[1][0].dispersion))

# The transition probability between two hidden states, for task-repetitions and switches seperately 

# Task-repetition 
print(np.exp(SHMM_3.model_transition[0].predict_log_proba(np.array([[0]]))))
print(np.exp(SHMM_3.model_transition[1].predict_log_proba(np.array([[0]])))) 
# Interpretation:
    # State 1 (fast): 96.4% chance of staying in fast resp. for task-repetitions
    # State 2 (slow): 70.7% chance of staying in slow resp. for task-repetitions
    ### Very high prob of staying in fast resp. makes sense in task-repetition, and considerably lower prob of
    ### Staying in slow resp. as well > switch towards slow resp. if instructions change > if instructions stay the
    ### same use (or switch to) fast resp. > prepare for long sequence of repetitions
    ### Also makes sense that prob is still pretty high in state 2 bc states tend to last for longer durations 
    
# Task-switch
print(np.exp(SHMM_3.model_transition[0].predict_log_proba(np.array([[1]])))) # 88.8% chance of staying in fast resp. 
print(np.exp(SHMM_3.model_transition[1].predict_log_proba(np.array([[1]])))) # 87.7% chance of staying in slow resp. 
# Interpretation:
    ## Makes sense that the chance of staying in fast-resp-mode is lower in task-switches
    ## Also makes sense that the chance of staying in slow-resp-mode is higher compared to task-repetitions
    
### Large probs. of stayinhg in the same state makes sense since states tend to persist for more trials and the 
### timescale of repetitions vs switches is on a trial-by-trial level

### Next model
# Use switch as emission covariate so we can study switch costs, and use reward and sequence as transition covariates
# since they are on the timescale of blocks (17 trials), i.e. levels switch per block/remain equal for 17 trials

# ---> First check which model performs best as sanity check

# =================================================================================================================

### Model comparison

# Dictionary to hold results for each model
results = {}

# Define models and data
models = {"Model1": (SHMM_1, df), "Model2": (SHMM_2, df), "Model3": (SHMM_3, df)}

# Loop through each model, calculate AIC and BIC, and store in results dictionary
for model_name, (SHMM, dataframe) in models.items():
    log_likelihood, num_params, num_observations = model_attributes(SHMM, dataframe)
    aic, bic = calculate_aic_bic(log_likelihood, num_params, num_observations)
    results[model_name] = [aic, bic]
    
results_df = pd.DataFrame(results, index=["AIC", "BIC"]).transpose() # Easier for subsequent step


min_aic_model = [results_df['AIC'].idxmin(), min(results_df['AIC'])]
min_bic_model = [results_df['BIC'].idxmin(), min(results_df['BIC'])]
print(min_aic_model, min_bic_model)

#  Model 2 wins > in line with what I thought
## Switch makes more sense as emissions covariate to investigate switch cost 
## Timescale (trial-by-trial) too short for impacting state switches

# =================================================================================================================

### Model 4 - Only RT, switch as emission covariate + sequence as transition covariate 
SHMM_4 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states


# This model has only one output which is modeled by a linear regression model
SHMM_4.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_4.set_inputs(covariates_initial = [], covariates_transition = ['Sequence'], covariates_emissions = [['Switch'], []])
SHMM_4.set_outputs([['RT']]) # we only have a list of one sequence.

random.seed(42)
SHMM_4.set_data([df]) # Specify dataframe
SHMM_4.train() # Run model

# GLM 
print(SHMM_4.model_emissions[0][0].coef) # Intercept for state 1, 1152 RT, 124 ms switch-cost > slow responding
print(SHMM_4.model_emissions[1][0].coef) # Intecept for state 2, 636 RT, 86 ms switch-cost > fast responding

print(np.sqrt(SHMM_4.model_emissions[0][0].dispersion))
print(np.sqrt(SHMM_4.model_emissions[1][0].dispersion))

 # The transition probability between two hidden states
 # Fake frequent 
print(np.exp(SHMM_4.model_transition[0].predict_log_proba(np.array([[0]])))) # 80.4% chance of staying in slow resp.
print(np.exp(SHMM_4.model_transition[1].predict_log_proba(np.array([[0]])))) # 92.3% chance of staying in fast resp.

# Fake rare
print(np.exp(SHMM_4.model_transition[0].predict_log_proba(np.array([[1]])))) # 80.1% chance of staying in slow resp.
print(np.exp(SHMM_4.model_transition[1].predict_log_proba(np.array([[1]])))) # 92.8% chance of staying in fast resp.

## No differences between fake sequences 

# No instruction
print(np.exp(SHMM_4.model_transition[0].predict_log_proba(np.array([[2]])))) # 79,7% chance of staying in slow resp.
print(np.exp(SHMM_4.model_transition[1].predict_log_proba(np.array([[2]])))) # 93.2% chance of staying in fast resp

## Difference very small with fake sequences, how about the real ones?

# Real frequent
print(np.exp(SHMM_4.model_transition[0].predict_log_proba(np.array([[3]])))) # 79,4% chance of staying in slow resp.
print(np.exp(SHMM_4.model_transition[1].predict_log_proba(np.array([[3]])))) # 93.7% chance of staying in fast resp

# Real rare
print(np.exp(SHMM_4.model_transition[0].predict_log_proba(np.array([[4]])))) # 79,1% chance of staying in slow resp.
print(np.exp(SHMM_4.model_transition[1].predict_log_proba(np.array([[4]])))) # 94% chance of staying in fast resp

## While all these effects are in the expected direction, differences are really minimal
## We can already deduce from this that sequence is not a relevant factor influencing state transitions 

# Maybe could be that sequence has better results when we only consider two sequences that are very different 
# Doesn't matter that much, just learning to use the package :)

# =================================================================================================================

### Model 5 - Only RT, switch as emission covariate + reward as transition covariate 
SHMM_5 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states


# This model has only one output which is modeled by a linear regression model
SHMM_5.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_5.set_inputs(covariates_initial = [], covariates_transition = ['Reward'], covariates_emissions = [['Switch'], []]) 
SHMM_5.set_outputs([['RT']]) # we only have a list of one sequence.

random.seed(42)
SHMM_5.set_data([df]) # Specify dataframe
SHMM_5.train() # Run model

print(SHMM_5.model_emissions[0][0].coef) # Intercept for state 1, 635 RT, 86 ms switch-cost > fast responding
print(SHMM_5.model_emissions[1][0].coef) # Intecept for state 2, 1153 RT, 125 ms switch-cost > slow responding

print(np.sqrt(SHMM_5.model_emissions[0][0].dispersion))
print(np.sqrt(SHMM_5.model_emissions[1][0].dispersion))

 # The transition probability between two hidden states
 # No Reward 
print(np.exp(SHMM_5.model_transition[0].predict_log_proba(np.array([[0]])))) # 91.1% chance of stayin in fast resp.
print(np.exp(SHMM_5.model_transition[1].predict_log_proba(np.array([[0]])))) # 80.8% chance of staying in slow resp.

# Reward 
print(np.exp(SHMM_5.model_transition[0].predict_log_proba(np.array([[1]])))) # 94.7% chance of stayin in fast resp.
print(np.exp(SHMM_5.model_transition[1].predict_log_proba(np.array([[1]])))) # 76.4% chance of staying in slow resp.

## Okay so this is in the direction that we expect, higher chance of staying in fast resp for reward 
## and lower chance of staying in slow resp. for reward
## Model makes more sense than sequence model 


# ================================================================================================================

### Model 6 -  Only RT, switch as emission covariate + sequence & reward as transition covariate

SHMM_6 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states


# This model has only one output which is modeled by a linear regression model
SHMM_6.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_6.set_inputs(covariates_initial = [], covariates_transition = ['Reward', 'Sequence'], covariates_emissions = [['Switch'], []]) 
SHMM_6.set_outputs([['RT']]) # we only have a list of one sequence.

random.seed(42)
SHMM_6.set_data([df]) # Specify dataframe
SHMM_6.train() # Run model

### Interpretation   
print(SHMM_6.model_emissions[0][0].coef) # Intecept for state 1, 1153 RT, 124 ms switch-cost > slow responding
print(SHMM_6.model_emissions[1][0].coef) # Intercept for state 2, 636 RT, 86 ms switch-cost > fast responding

### The transition probability between two hidden states

## Use both variables to calculate transition matrix > cumbersome to write out because 5 levels for sequence
print(np.exp(SHMM_6.model_transition[0].predict_log_proba(np.array([[0, 0]])))) # 81.5% staying in slow resp.
print(np.exp(SHMM_6.model_transition[1].predict_log_proba(np.array([[0, 0]]))))  # 89.9% staying in fast resp.

print(np.exp(SHMM_6.model_transition[0].predict_log_proba(np.array([[1, 0]]))))  # 77.2% staying in slow resp.
print(np.exp(SHMM_6.model_transition[1].predict_log_proba(np.array([[1, 0]]))))  # 94.0%  staying in fast resp.
# ===============================================================================================================

### Model comparison

# Dictionary to hold results for each model
results = {}

# Define models and data as needed (use your own model and dataframe names)
models = {"Model2": (SHMM_2, df), "Model4": (SHMM_4, df), "Model5": (SHMM_5, df), "Model6": (SHMM_6, df)}

# Loop through each model, calculate AIC and BIC, and store in results dictionary
for model_name, (SHMM, dataframe) in models.items():
    log_likelihood, num_params, num_observations = model_attributes(SHMM, dataframe)
    aic, bic = calculate_aic_bic(log_likelihood, num_params, num_observations)
    results[model_name] = [aic, bic]
    
results_df = pd.DataFrame(results, index=["AIC", "BIC"]).transpose() # Easier for subsequent step


min_aic_model = [results_df['AIC'].idxmin(), min(results_df['AIC'])]
min_bic_model = [results_df['BIC'].idxmin(), min(results_df['BIC'])]
print(min_aic_model, min_bic_model) # AIC prefers model 6, BIC prefers model 5
## We prefer the BIC, since it gives a larger penalty on extra parameters + pick most parsimonous model

# We stick with model 5, and remove sequence as transition covariate 
# Want to test if extra emission covariates further improve the model tho 

# =================================================================================================================

SHMM_7 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states


# This model has only one output which is modeled by a linear regression model
SHMM_7.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_7.set_inputs(covariates_initial = [], covariates_transition = ['Reward'], covariates_emissions = [['Switch', 'ACC'], []]) 
SHMM_7.set_outputs([['RT']]) # we only have a list of one sequence.

random.seed(42)
SHMM_7.set_data([df]) # Specify dataframe
SHMM_7.train() # Run model

### Interpretation
print(SHMM_7.model_emissions[0][0].coef) # 1056 RT, 125 ms switch-cost, 102 ms post-error slwoing >> slow responding
print(SHMM_7.model_emissions[1][0].coef) # 581 RT, 87 ms switch-cost, 56 ms post-error slwoing >> fast responding


### The transition probability between two hidden states

## No reward 
print(np.exp(SHMM_7.model_transition[0].predict_log_proba(np.array([[0]])))) # 80.6% staying in slow resp.
print(np.exp(SHMM_7.model_transition[1].predict_log_proba(np.array([[0]]))))  # 91.0% staying in fast resp.

## Reward 
print(np.exp(SHMM_7.model_transition[0].predict_log_proba(np.array([[1]])))) # 76.1% staying in slow resp.
print(np.exp(SHMM_7.model_transition[1].predict_log_proba(np.array([[1]]))))  # 94.7% staying in fast resp.

## Higher prob. of staying highly engaged when reward is avalaible, lower for slow response mode

# =================================================================================================================

SHMM_8 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states


# This model has only one output which is modeled by a linear regression model
SHMM_8.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_8.set_inputs(covariates_initial = [], covariates_transition = ['Reward'], covariates_emissions = [['Switch', 'Congruence'], []]) 
SHMM_8.set_outputs([['RT']]) # we only have a list of one sequence.

random.seed(42)
SHMM_8.set_data([df]) # Specify dataframe
SHMM_8.train() # Run model

### Interpretation
print(SHMM_8.model_emissions[0][0].coef) # 1195 RT, 124 ms switch-cost, 92 ms faster congruent trials >> slow responding
print(SHMM_8.model_emissions[1][0].coef) # 674 RT, 85 ms switch-cost, 77 ms faster congruent trials >> fast responding


### The transition probability between two hidden states

## No reward 
print(np.exp(SHMM_8.model_transition[0].predict_log_proba(np.array([[0]])))) # 81.0% staying in slow resp.
print(np.exp(SHMM_8.model_transition[1].predict_log_proba(np.array([[0]]))))  # 91.0% staying in fast resp.

## Reward 
print(np.exp(SHMM_8.model_transition[0].predict_log_proba(np.array([[1]])))) # 76.8% staying in slow resp.
print(np.exp(SHMM_8.model_transition[1].predict_log_proba(np.array([[1]]))))  # 94.7% staying in fast resp.

## Higher prob. of staying highly engaged when reward is avalaible, lower for slow response mode

# =================================================================================================================

SHMM_9 = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6) # 2 states

# This model has only one output which is modeled by a linear regression model
SHMM_9.set_models(model_emissions = [OLS()], # Linear regression
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

SHMM_9.set_inputs(covariates_initial = [], covariates_transition = ['Reward'], covariates_emissions = [['Switch', 'Congruence', 'ACC'], []]) 
SHMM_9.set_outputs([['RT']]) # we only have a list of one sequence.

random.seed(42)
SHMM_9.set_data([df]) # Specify dataframe
SHMM_9.train() # Run model

### Interpretation
print(SHMM_9.model_emissions[0][0].coef) # 588 rt, 87 ms switch-cost, 83 ms faster congruent trials AND
# 92 ms post-error slowing >> fast responding
print(SHMM_9.model_emissions[1][0].coef) # 1069 RT, 125 ms switch-cost, 97 ms faster congruent trials AND
# 132 ms post-error slwowing >> slow responding

### Initial state distribution 
print(np.exp(SHMM_9.model_initial.predict_log_proba(np.array([[]])))) # 99.9% chance of starting in slow modus
# Could also be because participants all start in 'fake frequent' blocks > expect frequent switches 

### The transition probability between two hidden states

## No reward 
print(np.exp(SHMM_9.model_transition[0].predict_log_proba(np.array([[0]])))) # 91.0 staying in fast resp.
print(np.exp(SHMM_9.model_transition[1].predict_log_proba(np.array([[0]]))))  # 90.7% staying in slow resp.

## Reward 
print(np.exp(SHMM_9.model_transition[0].predict_log_proba(np.array([[1]])))) # 94.6% staying in fast resp.
print(np.exp(SHMM_9.model_transition[1].predict_log_proba(np.array([[1]]))))  # 76.3% staying in slow resp.

## Higher prob. of staying highly engaged when reward is avalaible, lower for slow response mode


# ==============================================================================================================

### Model comparison

# Dictionary to hold results for each model
results = {}

# Define models and data as needed (use your own model and dataframe names)
models = {"Model5": (SHMM_5, df), "Model7": (SHMM_7, df), "Model8": (SHMM_8, df), 
          "Model9": (SHMM_9, df)}

# Loop through each model, calculate AIC and BIC, and store in results dictionary
for model_name, (SHMM, dataframe) in models.items():
    log_likelihood, num_params, num_observations = model_attributes(SHMM, dataframe)
    aic, bic = calculate_aic_bic(log_likelihood, num_params, num_observations)
    results[model_name] = [aic, bic]
    
results_df = pd.DataFrame(results, index=["AIC", "BIC"]).transpose() # Easier for subsequent step


min_aic_model = [results_df['AIC'].idxmin(), min(results_df['AIC'])]
min_bic_model = [results_df['BIC'].idxmin(), min(results_df['BIC'])]
print(min_aic_model, min_bic_model)

### Model 9, with 3 emission covariates wins

# ===============================================================================================================

### Fit winning model per participant

import copy

# Dictionary to store results for each participant
participant_results = {}

# List of unique participant IDs
participant_ids = df["Subj"].unique()

random.seed(42)
# Loop through each participant
for ID in participant_ids:
    # Filter the data for the current participant
    participant_data = df[df["Subj"] == ID]

    # Clone the winning model for this participant
    model = copy.deepcopy(SHMM_9)

    # Fit the cloned model to the current participant's data
    model.set_data([participant_data])
    model.train()

    # Store the results, accessing parameters manually
    participant_results[ID] = {
        "fitted_model": model,
        "log_likelihood": model.log_likelihood,
        "model_params": {
            "model_emissions": model.model_emissions,
            "model_initial": model.model_initial,
            "model_transition": model.model_transition
        }
    }

participant_results # Fit is succesfull, but parameters are stored as SHMM objects 

### Extract parameter values 

# Define lists to store extracted parameters
data = {
    'participant': [],
    'log_likelihood': [],
    'emission_coefficients_s1': [],
    'emission_coefficients_s2': [],
    'initial_coefficients': [],
    'transition_probabilities_s1': [],
    'transition_probabilities_s2': []
}

# Iterate through each participant's results and extract the desired data
for participant, results in participant_results.items():
    data['participant'].append(participant)
    data['log_likelihood'].append(results['log_likelihood'])

    # Extract models from results
    emission_model = results['model_params']['model_emissions']
    initial_model = results['model_params']['model_initial']
    transition_model = results['model_params']['model_transition']

    
    # Extract emission coeffecients
    emission_coefficients_s1 = [emission_model[0][0].coef]
    emission_coefficients_s2 = [emission_model[1][0].coef]
        
    # Extract initial state distribution
    initial_coefficients = [np.exp(initial_model.predict_log_proba(np.array([[]])))]

    # Extract transition probabilities
    transition_probabilities_s1 = [
    np.exp(transition_model[0].predict_log_proba(np.array([[0]]))),
    np.exp(transition_model[0].predict_log_proba(np.array([[1]])))
        ]

    transition_probabilities_s2 = [
    np.exp(transition_model[1].predict_log_proba(np.array([[0]]))),
    np.exp(transition_model[1].predict_log_proba(np.array([[1]])))
        ]
    
    # Append extracted data to the corresponding lists
    data['emission_coefficients_s1'].append(emission_coefficients_s1)
    data['emission_coefficients_s2'].append(emission_coefficients_s2)
    data['initial_coefficients'].append(initial_coefficients)
    data['transition_probabilities_s1'].append(transition_probabilities_s1)
    data['transition_probabilities_s2'].append(transition_probabilities_s2)

# Convert to DataFrame
df_results = pd.DataFrame(data)
df_results

subj_fit_LL = df_results['log_likelihood'].sum() # Total log likelihood for the fit 
subj_fit_LL # -26711
model_attributes(SHMM_9, df) # -273809 for global fit > individuals fits hgive better results
subj_fit_LL > model_attributes(SHMM_9, df)[0] # Confirm


# ===============================================================================================================
# ===============================================================================================================

### Write loop that cross-validates number of states 

# Specify covariates winning model
transition_covariates = ['Reward']
emission_covariates = [['Switch', 'Congruence', 'ACC'], []]  # Emission covariates for two states
output = [['RT']]

# List of unique participant IDs
participant_ids = df["Subj"].unique()

# Define the range of states to cross-validate
max_states = 6

# Dictionary to store cross-validation results
cv_results = []

## Note: Running time ~30 min

# Loop over the number of states to test
for num_states in range(1, max_states + 1):
    print(f"Testing {num_states} states...")
    
    # Loop through each participant
    for ID in participant_ids:
        # Filter the data for the current participant
        participant_data = df[df["Subj"] == ID]
        
        # Initialize the model with the current number of states
        model = UnSupervisedIOHMM(num_states=num_states, max_EM_iter=200, EM_tol=1e-6)
        
        # Set the models for emissions, transitions, and initial distribution
        model.set_models(
            model_emissions=[OLS()] * num_states,  # List of OLS models for each state
            model_transition=CrossEntropyMNL(solver='lbfgs'),
            model_initial=CrossEntropyMNL(solver='lbfgs')
        )
        
        # Set covariates and outputs as in the optimal model SHMM_9
        model.set_inputs(
            covariates_initial=[],
            covariates_transition=transition_covariates,
            covariates_emissions=emission_covariates
        )
        model.set_outputs(output)
        
        # Clone the model to avoid contamination of parameters between participants
        model_clone = copy.deepcopy(model)
        
        # Fit the cloned model to the current participant's data
        model_clone.set_data([participant_data])
        model_clone.train()
        
        # Store results for the current participant and number of states
        cv_results.append({
            "participant": ID,
            "num_states": num_states,
            "log_likelihood": model_clone.log_likelihood,
            "model_params": {
                "model_emissions": model_clone.model_emissions,
                "model_initial": model_clone.model_initial,
                "model_transition": model_clone.model_transition
            }
        })

cv_results # ran succesfully, extract model parameters once more

# ==============================================================================================================

## Extract model parameters

# Initialize dictionary for storing extracted parameters
cv_data = {
    'participant': [],
    'num_states': [],
    'log_likelihood': [],
    'initial_coefficients': []
}

# Prepare to dynamically add emission and transition coefficients
for s in range(1, max_states + 1):
    cv_data[f'emission_coefficients_s{s}'] = []
    cv_data[f'transition_probabilities_s{s}'] = []
    

# Iterate through each result in cv_results to extract data
for result in cv_results:
    # Append participant ID, num_states, and log likelihood to cv_data
    cv_data['participant'].append(result['participant'])
    cv_data['num_states'].append(result['num_states'])
    cv_data['log_likelihood'].append(result['log_likelihood'])

    # Extract initial coefficients
    initial_model = result['model_params']['model_initial']
    initial_coefficients = np.exp(initial_model.predict_log_proba(np.array([[]])))
    cv_data['initial_coefficients'].append(initial_coefficients)

    # Extract emission coefficients for each state, up to `num_states`
    emission_model = result['model_params']['model_emissions']
    for s in range(1, result['num_states'] + 1):
        emission_coef = emission_model[s-1][0].coef if s <= len(emission_model) else None
        cv_data[f'emission_coefficients_s{s}'].append(emission_coef)

    # If num_states < max_states, add None to remaining emission coefficient fields for consistency
    for s in range(result['num_states'] + 1, max_states + 1):
        cv_data[f'emission_coefficients_s{s}'].append(None)

    # Extract transition probabilities for each state, up to `num_states`
    transition_model = result['model_params']['model_transition']
    for s in range(1, result['num_states'] + 1):
        transition_probs = [
            np.exp(transition_model[s-1].predict_log_proba(np.array([[0]]))),  # No-reward
            np.exp(transition_model[s-1].predict_log_proba(np.array([[1]])))   # Reward
        ]
        cv_data[f'transition_probabilities_s{s}'].append(transition_probs)

    # If num_states < max_states, add None to remaining transition probability fields
    for s in range(result['num_states'] + 1, max_states + 1):
        cv_data[f'transition_probabilities_s{s}'].append(None)

# Convert cv_data to DataFrame
df_cv_results = pd.DataFrame(cv_data)
df_cv_results

# ===============================================================================================================
# ===============================================================================================================

### Model compairson

# Sum log-likelihoods for each number of states
likelihoods = df_cv_results.groupby('num_states')['log_likelihood'].sum()
winning_model = likelihoods.idxmax() # idxmax because LL are negative > closest to 0 = best
print("The best performing model has:", winning_model, "states!") # 6 states wins

# This is purely based on log-likelihood however, no correction for extra parameters

### calculate AIC and BIC

# Initialize dictionaries to store the results
cv_model_attributes = {
    'num_states': [],
    'total_log_likelihood': [],
    'num_params': [],
    'n_observations': len(df)
}

aic_bic_results = {
    'num_states': [],
    'aic': [],
    'bic': []
}

# Loop through different numbers of states (from 1 to max_states)
for num_states in range(1, max_states + 1):
    # Calculate total log-likelihood for each num_states in cv_results
    total_log_likelihood = df_cv_results[df_cv_results['num_states'] == num_states]['log_likelihood'].sum()

    # Initialize a sample model to calculate model parameters
    sample_model = UnSupervisedIOHMM(num_states=num_states)
    sample_model.set_models(
        model_emissions=[OLS()] * num_states,
        model_transition=CrossEntropyMNL(solver='lbfgs'),
        model_initial=CrossEntropyMNL(solver='lbfgs')
    )
    sample_model.set_inputs(
        covariates_initial=[],
        covariates_transition=transition_covariates,
        covariates_emissions=emission_covariates
    )

    # Use model_attributes function to get log-likelihood and number of parameters for this model
    log_likelihood, num_params, _ = model_attributes(sample_model, df)

    # Store the model attributes for this num_states in the dictionary
    cv_model_attributes['num_states'].append(num_states)
    cv_model_attributes['total_log_likelihood'].append(total_log_likelihood)
    cv_model_attributes['num_params'].append(num_params)

    # Calculate AIC and BIC for this num_states model
    aic, bic = calculate_aic_bic(total_log_likelihood, num_params, cv_model_attributes['n_observations'])

    # Store AIC and BIC in the results dictionary
    aic_bic_results['num_states'].append(num_states)
    aic_bic_results['aic'].append(aic)
    aic_bic_results['bic'].append(bic)

# Convert results to DataFrames
df_cv_model_attributes = pd.DataFrame(cv_model_attributes)
df_aic_bic_results = pd.DataFrame(aic_bic_results)

df_cv_model_attributes, df_aic_bic_results # Model 6 also wins when accounting for number of parameters 

# ===============================================================================================================
# ===============================================================================================================

# Store models seperately 

### Lastly, make different df's for different models 
cv_mod_s1 = df_cv_results[df_cv_results['num_states'] == 1].dropna(axis = 1)
cv_mod_s2 = df_cv_results[df_cv_results['num_states'] == 2].dropna(axis = 1)
cv_mod_s3 = df_cv_results[df_cv_results['num_states'] == 3].dropna(axis = 1)
cv_mod_s4 = df_cv_results[df_cv_results['num_states'] == 4].dropna(axis = 1)
cv_mod_s5 = df_cv_results[df_cv_results['num_states'] == 5].dropna(axis = 1)
cv_mod_s6 = df_cv_results[df_cv_results['num_states'] == 6].dropna(axis = 1)

cv_mod_s6


cv_models = {} # I don't want no dictionairy 

for s in range(1, max_states + 1): # Loop through different number of states
    cv_mod = df_cv_results[df_cv_results['num_states'] == s].dropna(axis = 1)
    cv_models[s] = cv_mod


### END

# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

