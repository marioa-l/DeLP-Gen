# <h1 align="center">DeLP-Gen: A DeLP Program Generator</h1>
<p align="center">
    <img width="450" height="450" src="https://github.com/marioa-l/DeLP-Gen/blob/main/background-remove.png?raw=true">
</p>

This project implement a principled approach to automatic generation of DeLP programs.
# Features

- Create a set of `strict rules, defeasible rules, facts and assumptions` that can be applied for the generation of arguments and defeaters.
- Offer a `parameterization` that allows obtaining DeLP programs with different metrics of `size` and `structural complexity`.
- Generate a `minimum set` of facts and assumptions for the generation of arguments.
- Generate `dialectical trees` for a subset of arguments
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

The value of the size and complexity metrics is shown for each of 3 cases of generated DeLP programs.

<div align="center">

Size

<img width="350" height="250" src="https://github.com/marioa-l/DeLP-Gen/blob/main/size.jpg?raw=true">

Complexity

<img width="350" height="250" src="https://github.com/marioa-l/DeLP-Gen/blob/main/complexity.jpg?raw=true">

Time (sec.)

<img width="350" height="250" src="https://github.com/marioa-l/DeLP-Gen/blob/main/time.jpg?raw=true">
</div>


# Tech

This generator is fully developed using `python` and the `Numpy` library.

# Licence

GPL-3.0 license