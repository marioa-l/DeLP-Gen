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

global dummyMode
#dummyMode=True
dummyMode=False

# Number of programs to generate for each value variation
global n_programs
n_programs = 10

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
- n_rules: Total number of rules (srules and drules).
- n_facts_pres: Number of facts and presumptions.
- mddl: Mean length of defeasible derivation of any argument.
- h: Mean maximum length of argumentation lines.
- t: Mean number of argument lines arising from an argument.
- tau: Mean number of dialectical trees.
"""
metrics=["n_rules",
       "n_facts_pres",
       "mddl",
       "h",
       "t",
       "tau"]

# The minimum value for each parameters
params_min = [1,0.1,0.1,0.1,1,1,1,1,1,1]

# The maximum value for each parameters (not inclusive)
params_max = [11,1,1,1,11,11,11,11,11,11]

# The parameter steps 
params_steps = [2,0.1,0.1,0.1,1,1,1,1,1,1]

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
            p_values = copy.copy(params_min)
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
            metrics = ComputeMetrics(dp + param + '/' + value + '/', 'metrics', dp + param + '/' + value + '/', '')
            n_programs = glob.glob(dp + param + "/" + value + "/*.delp")
            metrics.compute_dataset(len(n_programs))


#def retrive_params(fp):
#    """
#    Given a filepath string, it returns a list of paramenter used in generating 
#    that delp program
#    """
#	return fp[fp.find("[")+1:fp.rfind("]")].split(",")
#
#
#def generate_csv(d,csv_fp):
#    """
#    Given the dictionary d_programs it computes the csv for each program. 
#    A filepath for csv is needed as input
#    """
#    with open(csv_fp, 'w') as f:
#    writer = csv.writer(f)	
#    writer.writerow(params+metrics+["time"])
#    for k in d.keys():
#	    for delp in d[k]:
#		    writer.writerow(retrive_params(delp)+method2(delp)+[str(method3(delp))])
#	f.close()
#
#
#def print_matrix_plot(labels,matrix,filepath):
#    """
#    ???
#    """
#	fig_cor, axes_cor = plt.subplots(1,1)
#	fig_cor.set_size_inches(6, 6)
#	myimage = axes_cor.imshow(matrix)
#	plt.colorbar(myimage)
#	axes_cor.set_xticks(np.arange(0,matrix.shape[0], matrix.shape[0]*1.0/len(labels)))
#	axes_cor.set_yticks(np.arange(0,matrix.shape[1], matrix.shape[1]*1.0/len(labels)))
#	axes_cor.set_xticklabels(labels)
#	axes_cor.set_yticklabels(labels)
#	plt.draw()
#	plt.savefig(filepath)


######
###### CREATING THE DATASET
######
#d=create_datasets()


######
###### GENERATING CSV
######
#generate_csv(create_datasets(), dir+"delpGeneratorMario.csv")


######
###### GENERATING PLOTS FOR STEP 1
######
#n_p=0
#for k in d.keys():
#	parameter_k=[x for x in range(params_min[n_p],params_max[n_p]+1,params_steps[n_p])]
#	data = {k: parameter_k}
#	for m in metrics:
#		data[m]=[]
#	data["time"]=[]
#	for delp in d[k]:
#		metrics_d_k=method2(delp)
#		for i in range(len(metrics_d_k)):
#			data[metrics[i]].append(metrics_d_k[i])
#		data["time"].append(method3(delp))
#	n_p=n_p+1
#
#	#
#	#PLOT
#	#
#
#	#DATAFRAME
#	labels=[k]+metrics+["time"]
#	df = pd.DataFrame(data,columns=labels)
#
#	#MATRIXES
#	matrix_pearson = df.corr(method='pearson')
#	#matrix_kendall = df.corr(method='kendall')
#	matrix_spearman = df.corr(method='spearman')
#	matrix_cov = df.cov()
#
#	#PRINT THE PLOTS FOR EACH PARAMETER
#	print_matrix_plot(labels,matrix_pearson,(dir+"plot_pearson_"+k+".png"))
#	#print_matrix_plot(labels,matrix_kendall,(dir+"plot_kendall_"+k+".png"))
#	print_matrix_plot(labels,matrix_spearman,(dir+"plot_spearman_"+k+".png"))
#	print_matrix_plot(labels,matrix_cov,(dir+"plot_cov_"+k+".png"))
#	
#	
#
#######
####### GENERATING PLOTS FOR STEP 2  (i.e., TIMES of the whole dataset over metrics)
#######
#for k in d.keys():
#	data = {m:[] for m in metrics}
#	data["time"]=[]
#	for delp in d[k]:
#		metrics_d_k=method2(delp)
#		for i in range(len(metrics_d_k)):
#			data[metrics[i]].append(metrics_d_k[i])
#		data["time"].append(method3(delp))
#
#
#	#PLOT
#	labels=metrics+["time"]
#	df = pd.DataFrame(data,columns=labels)
#	
#	#MATRIXES
#	matrix_pearson = df.corr(method='pearson')
#	#matrix_kendall = df.corr(method='kendall')
#	matrix_spearman = df.corr(method='spearman')
#	matrix_cov = df.cov()
#
#
#	#PRINT THE PLOTS FOR EACH PARAMETER
#	print_matrix_plot(labels,matrix_pearson,(dir+"plot_time_pearson_"+k+".png"))
#	#print_matrix_plot(labels,matrix_kendall,(dir+"plot_time_kendall_"+k+".png"))
#	print_matrix_plot(labels,matrix_spearman,(dir+"plot_time_spearman_"+k+".png"))
#	print_matrix_plot(labels,matrix_cov,(dir+"plot_time_cov_"+k+".png"))

# To test
dp = '../dpgtest/'
#create_datasets(dp)
#compute_metrics(dp)
