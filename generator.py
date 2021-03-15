import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
from utils import *
import json

"""
Hyperparams

# Size of the KB Base
KBBASE_SIZE
        
# Probability to Facts elements in the KB BASE
FACT_PROB

# Probability of negated literals in the KB BASE
NEG_PROB

# Probability of Defeasible Rules in KB
DRULE_PROB

# Max rules defining the same literal 
(MAX_RULES_PER_HEAD <= MAX_ARG_LEVEL)
MAX_RULESPERHEAD

# Max size of body argument
MAX_BODYSIZE

# Min number of different arguments in each level
MIN_ARGSLEVEL

# Max number of arguments in each level
MAX_ARGSLEVEL

# Levels of the KB
LEVELS

# Ramification factor for each dialectical tree
(Max number of defeater for each argument)
RAMIFICATION

# Max height of dialectical trees
TREE_HEIGHT

# Probability of attack a inner point of an argument
INNER_PROB

# Number of programs to generate
N_PROGRAMS
"""

class Generator:

    # Symbols
    srule_symbol = '<-'
    drule_symbol = '-<'
    
    #Utils
    utils = Utils()

    def __init__(self) -> None:
        """
        Constructor
        """
        
        """
        Define default values for all hyperparameters
        """ 
        self.params = Map({
                "KBBASE_SIZE": 5,
                "FACT_PROB": 0.5,
                "NEG_PROB": 0.5,
                "DRULE_PROB": 0.5,
                "MAX_RULESPERHEAD": 1,
                "MAX_BODYSIZE": 1,
                "MIN_ARGSLEVEL": 1,
                "MAX_ARGSLEVEL": 1,
                "LEVELS": 1,
                "RAMIFICATION": 1,
                "TREE_HEIGHT": 1,
                "INNER_PROB": 0.5,
                "N_PROGRAMS": 1
                })

        """
        Global values
        """
        # List to control rules defining the same literal
        # [Literal] = Literal to define one rule
        self.LITERALS = []
        # Index of literals
        self.COUNT_LIT = 0
        """"""
        
        """
        Structure to save all rules
        """
        # The base of presumptions and facts
        self.KB_BASE = {
            'presumptions': [],
            'facts': []
        }

        # The KB (program)
        self.KB = {}
        """"""


    def define_hyperparams(self, hyperparams: dict) -> None:
        self.params = Map(hyperparams)


    def clear_datastructures(self) -> None:
        """
        Clear all data structures and global values
        """
        self.LITERALS = []
        self.COUNT_LIT = 0
        self.KB_BASE['presumptions'] = []
        self.KB_BASE['facts'] = []
        self.KB = {}


    def create_strict_rule(self, head: str, body: list) -> str:
        """
        Create a strict rule.
        Args:
            -head: A literal (the head of the stric rule)
            -body: A list of literals (the body of the strict rule)
        """
        body_string = ','.join(body)
        return str(head + ' ' + self.srule_symbol + ' ' + body_string + '.')


    def create_def_rule(self, head: str, body: str) -> str:
        """
        Create a defeasible rule.
        Args:
            -head: A literal (the head of the defeasible rule)
            -body: A list of literals (the body of the defeasible rule)
        """
        body_string = ','.join(body)
        return str(head + ' ' + self.drule_symbol + ' ' + body_string + '.')


    def get_one_rule_level(self, level: int) -> str:
        """
        Select a head rule (literal) from a particular level
        Args:
            -level: A KB level
        """
        random = self.utils.get_random()
        if random < self.params.DRULE_PROB and len(self.KB[level]['drules']) != 0:
            index = self.utils.get_choice(len(self.KB[level]['drules']))
            return self.KB[level]['drules'][index][0]
        elif len(self.KB[level]['srules']) != 0:
            index = self.utils.get_choice(len(self.KB[level]['srules']))
            return self.KB[level]['srules'][index][0]
        else:
            index = self.utils.get_choice(len(self.KB[level]['drules']))
            return self.KB[level]['drules'][index][0]


    def get_one_rule_all_levels(self, top_level: int) -> str:
        """
        Select a head rule (literal) from a level in [0, top_level]
        Args:
            -top_level: A KB level
        """
        select_from = self.utils.get_randint(0, top_level)
        return get_one_rule_level(select_from)


    def build_body(self, level: int) -> list:
        """
        Build a list of literals whit at least one litera from
        a particular KB level.
        Args:
            -level: A KB level
        """
        body = []
        body_size = self.utils.get_randint(1, self.params.MAX_BODYSIZE + 1)
        body.append(self.get_one_rule_level(level - 1))

        for aux in range(body_size - 1):
            body.append(self.get_one_rule_all_levels(level))

        return body


    def build_body_def(self, body_dim: int) -> list:
        """
        PENDING!!!
        Build a list of literals with len body_dim
        Args:
            -body_dim: Size of the body (number of literals)
        """ 
        body = []
        body_size = self.utils.get_randint(1, self.params.MAX_BODYSIZE + 1)

        for aux in range(body_size):
            self.COUNT_LIT += 1
            #random_FP = np.random.random()
            random_FP = 1
            polarity = self.utils.get_random()

            index = str(self.COUNT_LIT)
            literal = ('~a_' + index if polarity < self.params.NEG_PROB else 
                                        'a_' + index)
            if random_FP < self.params.FACT_PROB:
                # New Fact
                self.KB_BASE['facts'].append(literal)
                self.KB[0]['srules'].append([literal, ['true']])
            else:
                # New Presumption
                self.KB_BASE['presumptions'].append(literal)
                self.KB[0]['drules'].append([literal, ['true']])
            body.append(literal)

        return body


    def build_arguments(self, level: int) -> None:
        """
        Build all arguments for a particular level
        Args:
            -level: A KB level
        """
        self.KB[level] = {'drules': [], 'srules': []}

        for aux in range(self.params.MIN_ARGSLEVEL):
            index = str(self.COUNT_LIT)
            polarity = self.utils.get_random()
            literal = ('~a_' + index if polarity < self.params.NEG_PROB else 
                        'a_' + index)
            rules_head = self.utils.get_randint(1, self.params.MAX_RULESPERHEAD + 1)
            self.LITERALS.extend([literal] * (rules_head - 1))
            body = self.build_body(level)
            random_DS = self.utils.get_random()
            if random_DS < self.params.DRULE_PROB:
                self.KB[level]['drules'].append([literal, body])
            else:
                self.KB[level]['srules'].append([literal, body])

            # To create the defeaters
            #print("Lit: " + literal + " Body: " + str(body)) 
            self.build_tree(literal, body, level, self.params.TREE_HEIGHT)

            self.COUNT_LIT += 1

        # To complete level
        max_level = self.utils.get_randint(self.params.MIN_ARGSLEVEL, 
                                            self.params.MAX_ARGSLEVEL + 1)
        for aux in range(max_level):
            if len(self.LITERALS) != 0:
                literal = self.utils.get_choice(self.LITERALS)
                self.LITERALS.remove(literal)
                body = self.build_body(level)
                random_DS = self.utils.get_random()
                if random_DS < self.params.DRULE_PROB:
                    self.KB[level]['drules'].append([literal, body])
                else:
                    self.KB[level]['srules'].append([literal, body])
            # To create the defeaters
            # print("To defeat: ", literal)
            # build_tree(literal, body, level, TREE_HEIGHT)
            else:
                # There are no more pending literals to construct rules
                index = str(self.COUNT_LIT)
                polarity = self.utils.get_random()
                literal = ('~a_' + index if polarity < self.params.NEG_PROB 
                            else 'a_' + index)
                rules_head = self.utils.get_randint(1, 
                                            self.params.MAX_RULESPERHEAD + 1)
                self.LITERALS.extend([literal] * (rules_head - 1))
                body = self.build_body(level)
                random_DS = self.utils.get_random()
                if random_DS < self.params.DRULE_PROB:
                    self.KB[level]['drules'].append([literal, body])
                else:
                    self.KB[level]['srules'].append([literal, body])
                # To create the defeaters
                # print("To defeat: ", literal)
                # build_tree(literal, body, level, TREE_HEIGHT)

                self.COUNT_LIT += 1


    def build_tree(self, literal: str, body: list, level: int, height: int) -> None:
        """
        Build the dialectical tree for a particular argument
        Args:
            literal: The conclusion of the root arguement
            body: The body of the root argument
            level: The KB level of the root argument
            height: The hight of the tree to be constructed
        """
        inners_point = [lit for lit in body if lit not in self.KB_BASE['facts']]
        if len(inners_point) != 0:
            # ramification is the max number of defeater for the actual argument
            ramification = self.utils.get_randint(1, self.params.RAMIFICATION + 1)
            if height == 0:
                # Tree leaves
                for aux in range(ramification):
                    random_def_point = self.utils.get_random()
                    if random_def_point < self.params.INNER_PROB:
                        # Build a defeater for some litearl in the body
                        inner_point = self.utils.get_choice(inners_point)
                        complement = self.utils.get_complement(inner_point)
                        # defeater_body = build_body(level)
                        defeater_body = self.build_body_def(len(body))
                        self.KB[level]['drules'].append([complement, defeater_body])
                        print("Lit: " + complement + " Body: " + str(defeater_body))
                    else:
                        # Build a defeater for literal
                        complement = self.utils.get_complement(literal)
                        # defeater_body = build_body(level)
                        defeater_body = self.build_body_def(len(body))
                        self.KB[level]['drules'].append([complement, defeater_body])
                        print("Lit: " + complement + " Body: " + str(defeater_body))
            else:
                # Internal levels of the dialectical tree
                defeaters = []
                for aux in range(ramification):
                    random_def_point = self.utils.get_random()
                    if random_def_point < self.params.INNER_PROB:
                        # Build a defeater for some litearl in the body
                        # and append in to defeaters list
                        inner_point = self.utils.get_choice(inners_point)
                        complement = self.utils.get_complement(inner_point)
                        # defeater_body = build_body(level)
                        defeater_body = self.build_body_def(len(body))
                        self.KB[level]['drules'].append([complement, defeater_body])
                        defeaters.append([complement, defeater_body])
                        print("Lit: " + complement + " Body: " + str(defeater_body))
                    else:
                        # Build a defeater for literal
                        # and append in to defeaters list
                        complement = self.utils.get_complement(literal)
                        # defeater_body = build_body(level)
                        defeater_body = self.build_body_def(len(body))
                        self.KB[level]['drules'].append([complement, defeater_body])
                        defeaters.append([complement, defeater_body])
                        print("Lit: " + complement + " Body: " + str(defeater_body))
                for defeater in defeaters:
                    self.build_tree(defeater[0], defeater[1], level, height - 1)
        else:
            pass


    def build_kb_base(self) -> None:
        """
        Build the KB Base (facts and presumptions only)
        """
        global COUNT_LIT, KB
        self.KB[0] = {'drules': [], 'srules': []}
        for i in range(self.params.KBBASE_SIZE):
            random_FP = self.utils.get_random()
            polarity = self.utils.get_random()

            index = str(self.COUNT_LIT)
            literal = ('~a_' + index if polarity < self.params.NEG_PROB else 
                                        'a_' + index)
            if random_FP < self.params.FACT_PROB:
                # New Fact
                self.KB_BASE['facts'].append(literal)
                self.KB[0]['srules'].append([literal, ['true']])
            else:
                # New Presumption
                self.KB_BASE['presumptions'].append(literal)
                self.KB[0]['drules'].append([literal, ['true']])

            self.COUNT_LIT += 1


    def build_kb(self, level: int) -> None:
        """
        Build the KB (program) from level 1 to <level>
        Args:
            -level: Total number of KB levels
        """
        if level == 1:
            self.build_arguments(level)
        else:
            self.build_kb(level - 1)
            self.build_arguments(level)


    def to_delp_format(self, result_path: str, id_program: int) -> None:
        """
        Save the delp program
        Args:
            result_path: The path for save the program
        """
        
        program = []
        to_string = ''
        for key, value in self.KB.items():
            kb_level = str(key)
            program.append('/*** KB LEVEL = ' + kb_level + ' ***/')
            for drule in value['drules']:
                rule = self.create_def_rule(drule[0], drule[1])
                program.append(rule)
            for srule in value['srules']:
                rule = self.create_strict_rule(srule[0], srule[1])
                program.append(rule)

        for rule in program:
            to_string += rule + '\n'

        with open(result_path + 'delp' + str(id_program) + '.delp', 'w') as outfile:
            outfile.write(to_string) 


    def generate(self, result_path: str, hyperparams = 'undefined') -> None:
        """
        Generate a delp program with hyperparams (if they are in <args>)
        Args:
            -result_path: The path to save the program
            -hyperparams: Hyperparams
        """
        if hyperparams != 'undefined':
            self.define_hyperparams(hyperparams)
        
        for id_program in range(self.params.N_PROGRAMS):
            self.clear_datastructures()
            self.build_kb_base()
            self.build_kb(self.params.LEVELS)
            self.to_delp_format(result_path, id_program) 
