#!/usr/bin/env python
"""
Script to generate the program dataset and analyze the relationship of the 
parameters and metrics of the DeLP Program Generator (DPG)

Some abbreviations:
    - drule: Defeasible Rule
    - srule: Strict Rule
    - head: Consequent of a rule (conclusion)
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import copy
import glob
from generator import Generator
from utils import *
from delpMetrics import ComputeMetrics
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr

"""
### DeLP Metrics ###
- base: Number of facts and presumptions
- rules: Number of defeasible and strict rules (except those of level 0).
- args: Number of arguments (except those of level 0).
- addl: Average Defeasible Derivation Length.
- t: Number of dialectical trees.
- b: Branching factor.
- h: Average heights of dialectical trees.
- time: Average time to compute all literals.
"""
metrics = ["base",
           "rules",
           "args",
           "addl",
           "t",
           "b",
           "h",
           "times"]

std_metrics = ["std_base",
               "std_rules",
               "std_args",
               "std_addl",
               "std_t",
               "std_b",
               "std_h",
               "std_times"]

"""
### DPG Parameters ###
- ext_seed: Minimum number of facts and presumptions.
- p_fact: Probability that an element is a fact.
- p_neg: Probability to create a negated atom.
- p_drule: Probability to create a defeasible rule (1 - p_drule to srule).
- max_argsconc: Maximum number of rules with the same head.
- max_bodysize: Maximum number of literals in the body of an argument.
- min_argslevel: Minimum number of distinct arguments in a level.
- levels: Program levels.
- defs: Maximum number of defeaters for an argument of the top level.
- depth_argline: Maximum height of dialectical trees.
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
    "TREE_HEIGHT"
]

# The minimum value for each parameter
params_min = [0, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]

# The maximum value for each parameter (not inclusive)
params_max = [220, 1.0, 1.0, 1.0, 6, 6, 6, 11, 6, 6]

# The parameter steps 
params_steps = [20, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]

#############################################
# The parameter values for non-variable param
#############################################
params_default_min = [50, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]

params_default_med = [150, 0.5, 0.5, 0.5, 3, 3, 3, 5, 3, 3]

params_default_max = [200, 0.9, 0.9, 0.9, 5, 5, 5, 10, 5, 5]

#############################################
# The parameter values for non-variable param (defined by authors)

#############################################
# "KBBASE_SIZE",
params_default_base = [None, 0.5, 0.5, 0.5, 3, 3, 3, 2, 1, 1]
# "FACT_PROB",
params_default_facts = [100, None, 0.5, 0.5, 3, 3, 3, 2, 3, 1]
# "NEG_PROB",
params_default_neg = [100, 0.5, None, 0.5, 3, 3, 3, 2, 1, 1]
# "DRULE_PROB",
params_default_drul = [100, 0.5, 0.5, None, 3, 3, 3, 2, 1, 1]
# "MAX_RULESPERHEAD",
params_default_heads = [100, 0.5, 0.5, 0.5, None, 3, 1, 3, 1, 1]
# "MAX_BODYSIZE",
params_default_body = [100, 0.5, 0.5, 0.5, 1, None, 3, 3, 1, 1]
# "MIN_ARGSLEVEL",
params_default_arglvl = [100, 0.5, 0.5, 0.5, 1, 3, None, 3, 1, 1]
# "LEVELS",
params_default_lvl = [100, 0.5, 0.5, 0.5, 1, 2, 2, None, 1, 1]
# "RAMIFICATION",
params_default_deft = [100, 0.5, 0.5, 0.5, 1, 5, 3, 2, None, 3]
# "TREE_HEIGHT"
params_default_height = [100, 0.5, 0.5, 0.5, 1, 5, 3, 2, 2, None]
# All defaults values
defaults_values = [
    params_default_base,
    params_default_facts,
    params_default_neg,
    params_default_drul,
    params_default_heads,
    params_default_body,
    params_default_arglvl,
    params_default_lvl,
    params_default_deft,
    params_default_height
]


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
    parameters_values = {params[i]: p_values[i] for i in range(len(p_values))}
    parameters_values['INNER_PROB'] = 0.0
    parameters_values['N_PROGRAMS'] = args.n
    parameters_values['PREF_CRITERION'] = "more_specific"
    with open(dp + '/parameters.json', 'w') as output:
        json.dump(parameters_values, output, indent=4)
    generator = Generator()
    params_to_gen = get_data_from_file(dp + '/parameters.json')
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
                         float(np.round(value, 1)) for value in variation]
            p_values = copy.copy(defaults_values[i])
            # if args.min:
            #    p_values = copy.copy(params_default_min)
            # elif args.med:
            #    p_values = copy.copy(params_default_med)
            # elif args.max:
            #    p_values = copy.copy(params_default_max)
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
        variations = os.listdir(dp + param + '/')
        for value in variations:
            dataset_path = dp + param + '/' + value + '/'
            metrics = ComputeMetrics(dataset_path, 'metrics', dataset_path, '')
            n_programs = glob.glob(dataset_path + "*.delp")
            metrics.compute_dataset(len(n_programs), False, False,
                                    100)
        print(param + ": Computed metrics")
    print("All metrics were computed")


def analyze_metrics(dp: str, parameter_directory: str, parameter: str):
    """
    Given the directory of a parameter, retrieve the metrics for each variation 
    to create a csv and then generate the correlation matrices
    Args:
        parameter_directory: The directory of the parameter to analyze its metrics.
        parameter: Parameter to analyze
    """
    variations = os.walk(parameter_directory)
    variations = sorted(next(variations)[1], key=string_to_int_float)
    csv_fp = dp + parameter + 'per_variations_metrics.csv'
    csv_files = []
    with open(csv_fp, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(params + metrics + std_metrics)
        for variation in variations:
            path = parameter_directory + variation + '/'
            load_params = json.load(open(path + 'parameters.json'))
            load_metrics = json.load(open(path + 'metrics.json'))
            value_params = [load_params[p] for p in params]
            value_metrics = [load_metrics[m]['mean'] for m in metrics if m != 'times']
            value_metrics.append(load_metrics['times']['mean'])
            std_value_metrics = [load_metrics[m]['std'] for m in metrics if m != 'times']
            std_value_metrics.append(load_metrics['times']['std'])
            writer.writerow(value_params + value_metrics + std_value_metrics)
            # To create the global csv
            n_programs = load_params['N_PROGRAMS']
            parameters_data = [value_params] * n_programs
            parameters_df = pd.DataFrame(parameters_data, columns=params)
            results_csv = pd.read_csv(path + 'variation_metrics.csv')
            csv_files.append(parameters_df.join(results_csv))
        f.close()
    csv_parameter = pd.concat(csv_files, ignore_index=True)
    aux_column = csv_parameter.pop('program')
    csv_parameter.insert(0, 'program', aux_column)
    csv_parameter.to_csv(dp + parameter + 'total_metrics.csv')
    
    # To draw and save correlation matrix
    
    #data_csv = pd.read_csv(dp + parameter + 'metrics_csv.csv')
    data_csv = pd.read_csv(dp + parameter + 'total_metrics.csv')
    p_csv = data_csv[[parameter] + metrics]
    labels = [parameter] + metrics
    
    # To create the dataframe of metrics and running time
    metrics_csv = data_csv[metrics]

    # Pearson Correlation
    matrix_pearson = round(p_csv.corr(method='pearson'), 2)
    #print_matrix_plot(labels, matrix_pearson, (dp + parameter + "plot_pearson_" + parameter + ".png"))
    corr_param = matrix_pearson[parameter]
    p_values = generate_pvalues_matrix(p_csv)
    p_value_param = p_values[parameter]
    
    # Kendall Correlation
    # matrix_kendall = round(p_csv.corr(method='kendall'), 2)
    # print_matrix_plot(labels, matrix_kendall, (dp + parameter + "plot_kendall_" + parameter + ".png"))

    # Spearman Correlation
    # matrix_spearman = round(p_csv.corr(method='spearman'), 2)
    # print_matrix_plot(labels, matrix_spearman, (dp + parameter + "plot_spearman_" + parameter + ".png"))

    # Cov Correlation
    # matrix_cov = round(p_csv.cov(), 2)
    # print_matrix_plot(labels, matrix_cov, (dp + parameter + "plot_cov_" + parameter + ".png"))
    return corr_param, p_value_param, metrics_csv


def generate_pvalues_matrix(df):
    dfcols = pd.DataFrame(columns=df.columns)
    values = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            corr_value = round(pearsonr(tmp[r], tmp[c])[0], 2)
            p_value = round(pearsonr(tmp[r], tmp[c])[1], 3)
            values[r][c] = [''.join(['*' for t in [.05, .01, .001] if p_value
                                    <= t]),p_value]
    return values


def nan_to_cero(number):
    if pd.isna(number):
        return 0.00
    else:
        return number


def print_matrix_plot(labels, matrix, filepath):
    fig_cor, axes_cor = plt.subplots(1, 1)
    fig_cor.set_size_inches(12, 12)
    mask = np.tri(matrix.shape[0], k=0)
    matrix = np.ma.array(matrix, mask=np.transpose(mask))  # mask out the lower triangle
    img = axes_cor.imshow(matrix, cmap=plt.cm.get_cmap('RdYlGn', 10), vmin=-1, vmax=1)
    plt.colorbar(img, fraction=0.046, pad=0.04)
    axes_cor.set_xticks(np.arange(0, matrix.shape[0], matrix.shape[0] * 1.0 / len(labels)))
    axes_cor.set_yticks(np.arange(0, matrix.shape[1], matrix.shape[1] * 1.0 / len(labels)))
    axes_cor.set_xticklabels(labels, rotation=45, ha="right")
    axes_cor.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = axes_cor.text(j, i, str(nan_to_cero(matrix[i, j])), ha="center", va="center", color="black", size=12,
                                 bbox={'facecolor': 'white'})

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def generate_correlations_matrix(dp, correlations, pvalues):
    params = correlations.columns
    metrics = correlations.index
    fig_cor, axes_cor = plt.subplots(1, 1)
    fig_cor.set_size_inches(12, 12)
    img = axes_cor.imshow(correlations, cmap=plt.cm.get_cmap('RdYlGn', 20),
                          vmin=-1, vmax=1)
    plt.title("Correlations", size=20, fontweight='bold')
    plt.xlabel("Parameters", size=18)
    plt.ylabel("Metrics", size=18)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(img, cax=cax, ticks=np.arange(-1,1.1,.1))
    axes_cor.set_xticks(np.arange(0, correlations.shape[1],
                                  correlations.shape[1] * 1.0 / len(params)))
    axes_cor.set_yticks(np.arange(0, correlations.shape[0],
                                  correlations.shape[0] * 1.0 / len(metrics)))
    axes_cor.set_xticklabels(params, rotation=45, ha="right", fontweight='bold')
    axes_cor.set_yticklabels(metrics, fontweight='bold')
    matrix = correlations.to_numpy()
    for i in range(len(metrics)):
        for j in range(len(params)):
            text = axes_cor.text(j, i, str(nan_to_cero(matrix[i, j])) +
                    pvalues.iloc[i][j][0] + "\n[" + str(pvalues.iloc[i][j][1]) +
                    ']', ha="center", va="center",
                                 color="black", size=10, bbox={'facecolor': 'white'})

    plt.savefig(dp + 'correlations.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_correlations_metrics_times(dp, metrics_df):
    metrics_df.to_csv(dp + 'metrics_times.csv')
    correlations = round(metrics_df.corr(method='pearson'),2).loc[['times'],['base','rules','args','addl','t','b','h']]
    pvalues = generate_pvalues_matrix(metrics_df).loc[['times'],['base','rules','args','addl','t','b','h']]
    params = correlations.columns
    metrics = correlations.index
    fig_cor, axes_cor = plt.subplots(1, 1)
    fig_cor.set_size_inches(12, 12)
    img = axes_cor.imshow(correlations, cmap=plt.cm.get_cmap('RdYlGn', 20),
                          vmin=-1, vmax=1)
    plt.title("Metrics - Running Time Correlation", size=20, fontweight='bold')
    plt.xlabel("Metrics", size=18)
    plt.ylabel("Times", size=18)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.9, pack_start=True)
    fig_cor.add_axes(cax)
    plt.colorbar(img, cax=cax, ticks=np.arange(-1,1.1,.1),
            orientation="horizontal")
    axes_cor.set_xticks(np.arange(0, correlations.shape[1],
                                  correlations.shape[1] * 1.0 / len(params)))
    axes_cor.set_yticks(np.arange(0, correlations.shape[0],
                                  correlations.shape[0] * 1.0 / len(metrics)))
    axes_cor.set_xticklabels(params, ha="right", fontweight='bold')
    axes_cor.set_yticklabels(metrics, fontweight='bold')
    matrix = correlations.to_numpy()
    for i in range(len(metrics)):
        for j in range(len(params)):
            text = axes_cor.text(j, i, str(nan_to_cero(matrix[i, j])) +
                    pvalues.iloc[i][j][0] + "\n[" + str(pvalues.iloc[i][j][1]) +
                    ']', ha="center", va="center",
                                 color="black", size=10, bbox={'facecolor': 'white'})

    plt.savefig(dp + 'metrics_times_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def remove_old_files(dp):
    fileList = glob.glob(dp + "*.csv")
    files = fileList + glob.glob(dp + "*.png")
    for filePath in files:
        try:
            os.remove(filePath)
        except e:
            print(Bcolors.WARNING + "Error while deleting file: ", filePath)


def analyze_corr(dp: str) -> None:
    remove_old_files(dp)
    # To analyze correlations
    parameters = os.listdir(dp)
    corr_params = pd.DataFrame()
    p_values = pd.DataFrame()
    metrics = []
    for parameter in parameters:
        print("Starting with parameter " + parameter)
        corr_param, p_value_param, metrics_csv = analyze_metrics(dp, dp + parameter + '/', parameter)
        corr_params[parameter] = corr_param.iloc[1:]
        p_values[parameter] = p_value_param.iloc[1:]
        metrics.append(metrics_csv)
        print("...complete")
    print("\nAll Complete")
    generate_correlations_matrix(dp, corr_params.loc[:, params], p_values.loc[:, params])
    metrics_csv = pd.concat(metrics)
    generate_correlations_metrics_times(dp, metrics_csv)


"""
Main
"""
parser = argparse.ArgumentParser(
    prog='DPG Script',
    description='Script to create datasets, compute and analyze metrics of DeLP ' \
                'programs generated by DPG',
    epilog='The path (-path) is the directory where the dataset will be generated ' \
           'and where the results will be saved. In case of computing or ' \
           'analyzing the metrics, path is where the DeLP programs are.'
)

parser.add_argument('-path',
                    type=str,
                    help='The dataset path',
                    required=True)
parser.add_argument('-all',
                    action='store_true',
                    help='To create dataset, compute and analyze its metrics')
parser.add_argument('-create',
                    action='store_true',
                    help='To create dataset')
parser.add_argument('-compute',
                    action='store_true',
                    help='To compute dataset metrics')
parser.add_argument('-analyze',
                    action='store_true',
                    help='To analyze metrics')
# parser.add_argument('-min',
#                    action='store_true',
#                    help='To use the minimum default values for parameters' \
#                         ' that do no vary in each configuration')
# parser.add_argument('-med',
#                    action='store_true',
#                    help='To use the medium default values for parameters' \
#                         ' that do no vary in each configuration')
# parser.add_argument('-max',
#                    action='store_true',
#                    help='To use the maximum default values for parameters' \
#                         ' that do no vary in each configuration')
parser.add_argument('-n',
                    type=int,
                    help='Number of programs to generate')

args = parser.parse_args()

if not os.path.isdir(args.path):
    print("The path specified does not exist")
if args.all:
    create_datasets(args.path)
    compute_metrics(args.path)
    analyze_corr(args.path)
if args.create:
    create_datasets(args.path)
if args.compute:
    compute_metrics(args.path)
if args.analyze:
    analyze_corr(args.path)
