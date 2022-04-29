# <h1 align="center">DeLP-Gen: A DeLP program generator</h1>
<p align="center">
    <img width="400" height="400" src="https://github.com/marioa-l/DeLP-Gen/blob/main/background-remove.png?raw=true">
</p>

This project implement a principled approach to automatic generation of DeLP programs.
# Features

# Design

- Bottom-up approch

It begins by generating the basic components on which the most complex structures will be created. First the `facts` and `presumptions`, which form the `base` of the program.

- Creation by levels

The `arguments` are organized according to a `level` in the program. This level is represented by an integer value that indicates the maximum number of rules that are used in its derivation chain until reaching a base element. In this way, an argument of level _N_ must have a rule of level _N-1_ and so on until reaching the base components (facts and presumptions).

- Defeaters for top-level arguments

`Dialectical trees` are generated for each `top-level argument` of the program.

# Parameters

| Param | Meaning |
|-------|---------|
| `BASE_SIZE`     | Minimum number of facts and presumptions to create arguments|
| `FACT_PROB`     | Probability that an element of the program base is a fact   |
| `NEG_PROB`      | Probability to create a negated atom                        |
| `DRULE_PROB`    | Probability to create a defeasible rule                     |
| `MAX_RULESPERHEAD` | Maximum number of rules with the same literal in their head |
| `MAX_BODYSIZE`     | Maximum number of literals in the body of an argument       |
| `LEVEL`            | Programs levels                                             |
| `RAMIFICATION`     | Maximum number of defeaters for an argument                 |
| `TREE_HEIGHT`      | Maximum height of dialectical trees                         |
| `INNER_PROB`       | Probability that the attack relationship is of the “internal” type |
| `MIN_ARGSLEVEL`    | Minimum number of distinct arguments at a level of the program |

# Test

# Tech

# Licence
