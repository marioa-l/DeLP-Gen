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
- `N_PROGRAMS` : Number of programs to generate

## Usage
DPG is equipped with a script that allows generating DeLP programs. The script is called [`main.py`](https://github.com/marioa-l/DeLP-Gen/blob/main/main.py):

usage: main.py [`-h`] [`-load` *path*]
- `-load` [*path*] The path to load the file with the parameters and save the results.

***
**Important**

The value of the parameters to generate the programs must be specified in a `parameters.json` file and left at the path where you want to generate the programs.
***


**Example**

Here are a example of how to use the script generator:

	main -load PATH

In `PATH` there should be the `parameters.json` file formed as follows:

	{
		"BE": 5,
		"FACTS": 0.5,
		"NEG_PROB": 0.5, (still development)
		"DRUL": 0.5,
		"HEADS": 1,
		"BODY": 2,
		"ARGLVL": 2,
		"LVL": 2,
		"DEFT": 1,
		"HEIGHT": 1,
		"INNER_PROB": 0.0, (still development)
		"N_PROGRAMS": 50,
		"PREF_CRITERION": "more_specific" (still development)
	}

This will generate in `PATH` 50 delp programs with the value of the specified parameters. The parameters that are under development are not considered in this implementation, but they must be specified in the `parameter.json` file.
