<div align="center">
  <img src="image.png" alt="DPG Icon" with="300" height="300">
  <h1>A DeLP Program Generator</h1>
</div>

## Overview

DPG (DeLP Program Generator) is a system designed to generate synthetic DeLP programs. Its primary goal is to create non-trivial structures that guarantee the existence of arguments, defeaters, and dialectical trees. This is achieved by allowing users to set various parameters, enabling the creation of programs with different sizes and structural complexities.

## Parameters

The following parameters can be adjusted to customize the generated DeLP programs:

- `BE` : Minimum number of base elements 
- `FACTS` : Percentage of facts in the base 
- `DRUL` : Percentage of defeasible rules 
- `HEADS` : Maximum number of rules with the same head literal 
- `BODY` : Maximum number of literals in the rule’s body 
- `ARGLVL` : Minimum number of distinct arguments for each level 
- `LVL` : Maximum argument level that can be reached 
- `DEFT` : Maximum number of defeaters for an argument 
- `HEIGHT` : Height of dialectical trees 

## Usage
DPG is equipped with a script that allows generating DeLP programs and computing the value of their metrics. Furthermore, the computation of metrics can be applied to a single program or a set of programs. The script is called [`main.py`](https://github.com/marioa-l/DeLP-Gen/blob/main/main.py) and accepts the following input arguments:

usage: main.py [`-h`] [`-load` *path*] [`-all`] [`-gen`] [`-compute`] [`-approx`] [`-perc` %] [`-one`] [`-p` *program*] [`-defs`] [`-gencsv`]

- `-load` [*path*] The path for loading the dataset and saving the results 
- `-all` Compute a dataset - generate dataset and compute the metrics
- `-gen` Only generate the programs 
- `-compute` Compute the metrics
- `-approx` Compute an approximation value of metrics)
- `-perc` [%] Percentage of literals to consult per level to approximate metrics values 
- `-one` Compute metrics for one program
- `-p` [*program path*] DeLP program path
- `-defs` Print arguments-defeaters info
- `-gencsv` Evaluate set of parameters from csv

***
**Important**

The value of the parameters to generate the programs must be specified in a `parameters.json` file and left at the path where you want to generate the programs.
***

<!--
## Examples

Here are a few examples of how to use the script generator:

1. To generate and compute metrics for a dataset:-->

