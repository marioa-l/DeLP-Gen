import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

"""
Default values for all hiperparameters
"""
# Probability to Facts elements in the KB BASE
FACT_PROB = 0.5

# Probability of negated literals in the KB BASE
NEG_PROB = 0.5

# Probability of Defeasible Rules in KB
DEF_RULE_PROB = 0.5

# Probability of attack a inner point of an argument
INN_POINT_PROB = 0.5

# Max size of body argument
MAX_BODY_SIZE = 1

# Min number of different arguments in each level
MIN_DIF_ARG_LEVEL = 1

# Max number of arguments in each level
MAX_ARG_LEVEL = 1

# Max rules defining the same literal 
# (MAX_RULES_PER_HEAD <= MAX_ARGUMENTS_PER_LEVEL)
MAX_RULES_PER_HEAD = 1

# Ramification factor for each dialectical tree
# (Max number of defeater for each argument)
RAMIFICATION = 2

# Max height of dialectical trees
TREE_HEIGHT = 1

# Levels of the KB
LEVELS = 1 
""""""


# List to control rules defining the same literal
# [Litearl...] = Literal to define one rule
LITERALS = []

# Index of literals
COUNT_LIT = 0

# Number of a generated program
n_program: int = 0

# Symbols
strict_rule_symbol = '<-'
def_rule_symbol = '-<'

# The base of presumptions and facts
KB_BASE = {
    'presumptions': [],
    'facts': []
}

# The KB
KB = {}


def create_strict_rule(head: str, body: list) -> str:
    """
    Create a strict rule.
    Args:
        -head: A literal (the head of the stric rule)
        -body: A list of literals (the body of the strict rule)
    """
    body_string = ','.join(body)
    return str(head + ' ' + strict_rule_symbol + ' ' + body_string + '.')


def create_def_rule(head: str, body: str) -> str:
    """
    Create a defeasible rule.
    Args:
        -head: A literal (the head of the defeasible rule)
        -body: A list of literals (the body of the defeasible rule)
    """
    body_string = ','.join(body)
    return str(head + ' ' + def_rule_symbol + ' ' + body_string + '.')


def get_one_rule_level(level):
    random_DS = np.random.random()
    if random_DS < DEF_RULE_PROB and len(KB[level]['drules']) != 0:
        index = np.random.choice(len(KB[level]['drules']), 1)[0]
        return KB[level]['drules'][index][0]
    elif len(KB[level]['srules']) != 0:
        index = np.random.choice(len(KB[level]['srules']), 1)[0]
        return KB[level]['srules'][index][0]
    else:
        index = np.random.choice(len(KB[level]['drules']), 1)[0]
        return KB[level]['drules'][index][0]


def get_one_rule_all_levels(top_level):
    select_from = np.random.randint(0, top_level, 1)[0]
    return get_one_rule_level(select_from)


def build_body(level):
    body = []
    body_size = np.random.randint(1, MAX_BODY_SIZE + 1, 1)[0]
    body.append(get_one_rule_level(level - 1))

    for aux in range(body_size - 1):
        body.append(get_one_rule_all_levels(level))

    return body


def build_body_def(body_dim):
    global COUNT_LIT
    body = []
    body_size = np.random.randint(1, MAX_BODY_SIZE + 1, 1)[0]

    for aux in range(body_size):
        COUNT_LIT += 1
        #random_FP = np.random.random()
        random_FP = 1
        polarity = np.random.random()

        index = str(COUNT_LIT)
        literal = '~a_' + index if polarity < NEG_PROB else 'a_' + index
        if random_FP < FACT_PROB:
            # New Fact
            KB_BASE['facts'].append(literal)
            KB[0]['srules'].append([literal, ['true']])
        else:
            # New Presumption
            KB_BASE['presumptions'].append(literal)
            KB[0]['drules'].append([literal, ['true']])
        body.append(literal)

    return body


def build_arguments(level):
    global COUNT_LIT, KB, LITERALS

    KB[level] = {'drules': [], 'srules': []}

    for aux in range(MIN_DIF_ARG_LEVEL):
        index = str(COUNT_LIT)
        polarity = np.random.random()
        literal = '~a_' + index if polarity < NEG_PROB else 'a_' + index
        rules_per_literal = np.random.randint(1, MAX_RULES_PER_HEAD + 1, 1)[0]
        LITERALS.extend([literal] * (rules_per_literal - 1))
        body = build_body(level)
        random_DS = np.random.random()
        if random_DS < DEF_RULE_PROB:
            KB[level]['drules'].append([literal, body])
        else:
            KB[level]['srules'].append([literal, body])

        # To create the defeaters
        print("Lit: " + literal + " Body: " + str(body)) 
        build_tree(literal, body, level, TREE_HEIGHT)

        COUNT_LIT += 1

    # To complete level
    max_level = np.random.randint(MIN_DIF_ARG_LEVEL, MAX_ARG_LEVEL + 1, 1)[0]
    for aux in range(max_level):
        if len(LITERALS) != 0:
            literal = np.random.choice(LITERALS, 1)[0]
            LITERALS.remove(literal)
            body = build_body(level)
            random_DS = np.random.random()
            if random_DS < DEF_RULE_PROB:
                KB[level]['drules'].append([literal, body])
            else:
                KB[level]['srules'].append([literal, body])
        # To create the defeaters
        # print("To defeat: ", literal)
        # build_tree(literal, body, level, TREE_HEIGHT)
        else:
            # There are no more pending literals to construct rules
            index = str(COUNT_LIT)
            polarity = np.random.random()
            literal = '~a_' + index if polarity < NEG_PROB else 'a_' + index
            rules_per_literal = np.random.randint(1, MAX_RULES_PER_HEAD + 1, 1)[0]
            LITERALS.extend([literal] * (rules_per_literal - 1))
            body = build_body(level)
            random_DS = np.random.random()
            if random_DS < DEF_RULE_PROB:
                KB[level]['drules'].append([literal, body])
            else:
                KB[level]['srules'].append([literal, body])
            # To create the defeaters
            # print("To defeat: ", literal)
            # build_tree(literal, body, level, TREE_HEIGHT)

            COUNT_LIT += 1


# Get the complement of a literal
def get_complement(literal):
    if '~' in literal:
        return literal.replace('~', '')
    else:
        return '~' + literal


# Build the dialectical tree for a particular argument
# literal (String): The conclusion of the root argument
# body (List of lists, each element is a rule): The body of the root argument
# height (Int): The height of the tree
def build_tree(literal, body, level, height):
    inners_point = [lit for lit in body if lit not in KB_BASE['facts']]
    if len(inners_point) != 0:
        # ramification is the max number of defeater for the actual argument
        ramification = np.random.randint(1, RAMIFICATION + 1, 1)[0]
        if height == 0:
            # Tree leaves
            for aux in range(ramification):
                random_def_point = np.random.random()
                if random_def_point < INN_POINT_PROB:
                    # Build a defeater for some litearl in the body
                    inner_point = np.random.choice(inners_point, 1)[0]
                    complement = get_complement(inner_point)
                    # defeater_body = build_body(level)
                    defeater_body = build_body_def(len(body))
                    KB[level]['drules'].append([complement, defeater_body])
                    print("Lit: " + complement + " Body: " + str(defeater_body))
                else:
                    # Build a defeater for literal
                    complement = get_complement(literal)
                    # defeater_body = build_body(level)
                    defeater_body = build_body_def(len(body))
                    KB[level]['drules'].append([complement, defeater_body])
                    print("Lit: " + complement + " Body: " + str(defeater_body))
        else:
            # Internal levels of the dialectical tree
            defeaters = []
            for aux in range(ramification):
                random_def_point = np.random.random()
                if random_def_point < INN_POINT_PROB:
                    # Build a defeater for some litearl in the body
                    # and append in to defeaters list
                    inner_point = np.random.choice(inners_point, 1)[0]
                    complement = get_complement(inner_point)
                    # defeater_body = build_body(level)
                    defeater_body = build_body_def(len(body))
                    KB[level]['drules'].append([complement, defeater_body])
                    defeaters.append([complement, defeater_body])
                    print("Lit: " + complement + " Body: " + str(defeater_body))
                else:
                    # Build a defeater for literal
                    # and append in to defeaters list
                    complement = get_complement(literal)
                    # defeater_body = build_body(level)
                    defeater_body = build_body_def(len(body))
                    KB[level]['drules'].append([complement, defeater_body])
                    defeaters.append([complement, defeater_body])
                    print("Lit: " + complement + " Body: " + str(defeater_body))
            for defeater in defeaters:
                build_tree(defeater[0], defeater[1], level, height - 1)
    else:
        pass


# Build the KB BASE:
# dim_kb_base (Int): Number of presumptions and facts (presumptions + facts)
def build_kb_base(dim_kb_base):
    global COUNT_LIT, KB
    KB[0] = {'drules': [], 'srules': []}
    for i in range(dim_kb_base):
        random_FP = np.random.random()
        polarity = np.random.random()

        index = str(COUNT_LIT)
        literal = '~a_' + index if polarity < NEG_PROB else 'a_' + index
        if random_FP < FACT_PROB:
            # New Fact
            KB_BASE['facts'].append(literal)
            KB[0]['srules'].append([literal, ['true']])
        else:
            # New Presumption
            KB_BASE['presumptions'].append(literal)
            KB[0]['drules'].append([literal, ['true']])

        COUNT_LIT += 1


def build_kb(level):
    if level == 1:
        build_arguments(level)
    else:
        build_kb(level - 1)
        build_arguments(level)


def to_delp_format(result_path):
    global n_program
    program = []
    to_string = ''
    for key, value in KB.items():
        kb_level = str(key)
        program.append('/*** KB LEVEL = ' + kb_level + ' ***/')
        for drule in value['drules']:
            rule = create_def_rule(drule[0], drule[1])
            program.append(rule)
        for srule in value['srules']:
            rule = create_strict_rule(srule[0], srule[1])
            program.append(rule)

    for rule in program:
        to_string += rule + '\n'

    with open(result_path + '/delp' + str(n_program) + '.delp', 'w') as outfile:
        outfile.write(to_string)
    n_program += 1


def main(result_path):
    build_kb_base(10)
    build_kb(LEVELS)
    to_delp_format(result_path)
    # print(KB)


def build_dataset(n_programs, result_path):
    global KB_BASE, KB, COUNT_LIT

    for aux in range(n_programs):
        COUNT_LIT = 0
        KB = {}
        KB_BASE = {
            'presumptions': [],
            'facts': []
        }
        main(result_path)

result_path = sys.argv[1]

build_dataset(1, result_path)
# main(result_path)
