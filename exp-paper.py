#!/usr/bin/env python
"""
Script to generate the program dataset and analyze the relationship of the 
parameters and metrics of the DeLP Program Generator (DPG)

Some abbreviations:
    - drule: Defeasible Rule
    - srule: Strict Rule
    - head: Consequent of a rule (conclusion)
"""

import random
import statistics as stat
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import copy
import glob
from generator import Generator
from utils import *
from delpMetrics import ComputeMetrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib import colors

global dummyMode
#dummyMode=True
dummyMode=False

# Number of programs to generate for each value variation
global n_programs
n_programs = 5

"""
### DPG Parameters ###
- KBBASE_SIZE: Minimum number of facts and presumptions.
- FACT_PROB: Probability that an element is a fact.
- NEG_PROB: Probability to create a negated atom.
- DRULE_PROB: Probability to create a defeasible rule (1 - DRULE_PROB to srule).
- MAX_RULESPERHEAD: Maximum number of rules with the same head.
- MAX_BODYSIZE: Maximum number of literals in the body of an argument.
- MIN_ARGSLEVEL: Minimum number of distinct arguments in a level.
- LEVELS: Program levels.
- RAMIFICATION: Maximum number of defeaters for an argument of the top level.
- TREE_HEIGHT: Maximum height of dialectical trees.
"""
params = [
       "KBBASE_SIZE",
       "FACT_PROB",
       "NEG_PROB", 
       "DRULE_PROB",
       "MAX_RULESPERHEAD", 
       "MAX_BODYSIZE", 
       "MIN_ARGSLEVEL", 
       "LEVELS",
       "RAMIFICATION",
       "TREE_HEIGHT"]

"""
### DeLP Metrics ###
- arguments: Number of arguments.
- rules: Total number of rules (srules and drules).
- facts_presum: Number of facts and presumptions.
- mddl: Mean length of defeasible derivation of any argument.
- h: Mean maximum length of argumentation lines.
- t: Mean number of argument lines arising from an argument.
- tau: Number of dialectical trees. (NOT IMPLEMENTED).
- t_min: Time to respond to the status of the "simplest literal" in the program.
- t_max: Time to respond to the status of the "most dificult" literal of the program.
- t_mean: Average time to respond to a program literal.
"""
metrics=["arguments",
       "rules",
       "fact_preum",
       "mddl",
       "h",
       "t",
       "times"]
#       "tau"]

# The minimum value for each parameters
params_min = [1,0.1,0.1,0.1,1,1,1,1,1,1]

# The maximum value for each parameters (not inclusive)
params_max = [10,1.0,1.0,1.0,5,5,5,5,5,5]

# The parameter steps 
params_steps = [1,0.1,0.1,0.1,1,1,1,1,1,1]

# The parameter values for non-variable param
params_default_min = [2,0.2,0.2,0.2,2,2,2,2,2,2]

params_default_med = [5,0.5,0.5,0.5,5,5,5,5,5,5]

params_default_max = [10,1,1,1,10,10,10,10,10,10]

#Utils
utils = Utils()

def generate_programs(dp: str, p_values: list) -> None:
    """
    Given a directory path and parameters, generate a number of programs in 
    that directory and with the specified parameters. Return the directory
    path on completion
    Args:
        dp: Directory Path
        p_values: List of parameters values
    """
    check_directory = os.path.isdir(dp)
    if not check_directory:
        os.mkdir(dp)
    parameters_values = {params[i]:p_values[i] for i in range(len(p_values))}
    parameters_values['INNER_PROB'] = 0.0
    parameters_values['N_PROGRAMS'] = n_programs
    parameters_values['PREF_CRITERION'] = "more_specific"
    with open(dp + '/parameters.json', 'w') as output:
        json.dump(parameters_values, output)
    generator = Generator()
    params_to_gen = utils.get_data_from_file(dp + '/parameters.json')
    generator.generate(dp + '/', params_to_gen)


def create_datasets(dp):
    """
    Build the dataset on a specific directory path
    Args:
        dp: Directory path to sava dataset
    """ 
    check_directory = os.path.isdir(dp)
    if not check_directory:
        print("The specified directory does not exist")
        sys.exit()
    else:
        for i in range(len(params)):
            param_to_variate = params[i]
            param_path = dp + param_to_variate
            os.mkdir(param_path)
            variation = list(np.arange(params_min[i], 
                            params_max[i],
                            params_steps[i]))
            variation = [int(value) if isinstance(value, np.integer) else 
                                float(np.round(value,1)) for value in variation]
            p_values = copy.copy(params_default_min)
            for value in variation:
                p_values[i] = value
                generate_programs(param_path + '/' + str(value), p_values)
        print("Dataset created")


def compute_metrics(dp: str) -> None:
    """
    Given a directory path of DeLP programs, compute and save the mean of its 
    exacts metrics
    Args:
        dp: DeLP filepath to compute its exact metrics
    """
    params_directory = os.listdir(dp)
    for param in params_directory:
        variations = os.listdir(dp + param  + '/')
        for value in variations:
            dataset_path = dp + param + '/' + value + '/'
            metrics = ComputeMetrics(dataset_path, 'metrics', dataset_path, '')
            n_programs = glob.glob(dataset_path + "*.delp")
            metrics.compute_dataset(len(n_programs),False)
        print(param + ": Computed metrics")
    print("All metrics were computed")


def analyze_metrics(parameter_directory: str, parameter: str) -> None:
    """
    Given the directory of a parameter, retrieve the metrics for each variation 
    to create a csv and then generate the correlation matrices
    Args:
        parameter_directory: Directory of a parameter
        parameter: Parameter to analyze
    """
    variations = os.walk(parameter_directory)
    csv_fp = parameter_directory + 'metrics_csv.csv'
    with open(csv_fp, 'w') as f:
        writer = csv.writer(f)	
        writer.writerow(params + metrics)
        for variation in next(variations)[1]:
            path = parameter_directory + variation + '/'
            load_params = json.load(open(path + 'parameters.json')) 
            load_metrics = json.load(open(path + 'metrics.json'))
            value_params = [load_params[p] for p in params]
            value_metrics = [load_metrics[m] for m in metrics if m != 'times']
            value_metrics.append(load_metrics['times']['mean'])
            writer.writerow(value_params + value_metrics)
        f.close()
    
    # To draw and save correlation matrix
    data_csv = pd.read_csv(parameter_directory + 'metrics_csv.csv')
    p_csv = data_csv[[parameter] + metrics]
    labels = [parameter] + metrics
    #MATRIXES
    matrix_pearson = p_csv.corr(method='pearson')
    #matrix_kendall = p_csv.corr(method='kendall')
    #matrix_spearman = p_csv.corr(method='spearman')
    #matrix_cov = p_csv.cov()
    
    #PRINT THE PLOTS FOR EACH PARAMETER
    print_matrix_plot(labels,matrix_pearson,(parameter_directory+"plot_pearson_"+parameter+".png"))
    #print_matrix_plot(labels,matrix_kendall,(parameter_directory+"plot_kendall_"+parameter+".png"))
    #print_matrix_plot(labels,matrix_spearman,(parameter_directory+"plot_spearman_"+parameter+".png"))
    #print_matrix_plot(labels,matrix_cov,(parameter_directory+"plot_cov_"+parameter+".png"))


def print_matrix_plot(labels,matrix,filepath):
    fig_cor, axes_cor = plt.subplots(1,1)
    fig_cor.set_size_inches(12, 12)
    #cmap = colors.ListedColormap(['red','white','green'])
    
    #bounds = [-0.99,-0.5,0.5,0.99]
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    #plt.figure(figsize=(12,12), facecolor='w',edgecolor='k')
    #sns.set(font_scale=1.2)
    #mask_1 = np.triu(np.ones_like(matrix, dtype=bool))
    #sns.heatmap(matrix_pearson, cmap=cmap,
    #            center=0,
    #            annot=True,
    #            fmt='.1g',
    #            mask=mask_1,
    #            norm=norm)
    
    #img = axes_cor.imshow(matrix, cmap=cmap, norm=norm)
    img = axes_cor.imshow(matrix, cmap=plt.cm.get_cmap('Greens', 10), vmin=-1, vmax=1)
    
    #aux = axes_cor[1].imshow(mask_1)
    # make a color bar
    
    #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-0.99, -0.5,0.5, 0.99])
    plt.colorbar(img)
    #plt.savefig('redwhite.png')
    
    axes_cor.set_xticks(np.arange(0,matrix.shape[0], matrix.shape[0]*1.0/len(labels)))
    axes_cor.set_yticks(np.arange(0,matrix.shape[1], matrix.shape[1]*1.0/len(labels)))
    axes_cor.set_xticklabels(labels)
    axes_cor.set_yticklabels(labels)
    plt.show()
    plt.draw()
    plt.savefig(filepath)


def run_exp(dp: str) -> None:
    """
    Main procedure: Generate and compute dataset of DeLP programs
    """
    # Create datasets
    #create_datasets(dp)
    # Compute metrics
    #compute_metrics(dp)
    
    # To analyze correlations
    parameters = os.listdir(dp)
    for parameter in parameters:
        print("Starting with parameter " + parameter)
        analyze_metrics(dp + parameter + '/', parameter)
        print("\n complete")
    print("All Complete")

# To test
dp = sys.argv[1] 
run_exp(dp)
