import copy
from io import SEEK_CUR
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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
        self.params = {
                "KBBASE_SIZE": 5,
                "FACT_PROB": 0.5,
                "NEG_PROB": 0.5,
                "DRULE_PROB": 0.5,
                "MAX_RULESPERHEAD": 1,
                "MAX_BODYSIZE": 1,
                "MIN_ARGSLEVEL": 1,
                "LEVELS": 1,
                "RAMIFICATION": 1,
                "TREE_HEIGHT": 1,
                "INNER_PROB": 0.5,
                "N_PROGRAMS": 1,
                "PREF_CRITERION": "more_specific"
                }

        """
        Global values and auxiliar structures
        """
        # Index of literals
        self.COUNT_LIT = 0
        # Used heads 
        self.USED_HEADS = ()
        """"""
        
        """
        Structure to save all rules
        """
        self.rule = {
                'drules': [],
                'srules': [],
                'facts': [],
                'presumptions': []
                } 

        # The KB (program)
        self.levels = {}
        """"""


    def define_hyperparams(self, hyperparams: dict) -> None:
        self.params = copy.copy(hyperparams)


    def clear_datastructures(self) -> None:
        """
        Clear all data structures and global values
        """
        self.LITERALS = []
        self.COUNT_LIT = 0
        self.USED_RULES = []
        self.rules = {
                'drules': [],
                'srules': [],
                'facts': [],
                'presumptions': []
                }
        self.levels = {}


    def get_new_id(self) -> str:
        """
        Return an id for literal and update the counter
        """
        id = str(self.COUNT_LIT)
        self.COUNT_LIT += 1
        return id


    def create_strict_rule(self, head: str, body: Union[list,tuple]) -> str:
        """
        Create a strict rule.
        Args:
            -head: A literal (the head of the stric rule)
            -body: A list of literals (the body of the strict rule)
        """ 
        if isinstance(body[0], str):
            # Is fact
            body_string = 'true'
        else:
            # Is rule
            body_literals = self.get_body_literals(body)
            body_string = ','.join(body_literals)

        return str(head + ' ' + self.srule_symbol + ' ' + body_string + '.')


    def create_def_rule(self, head: str, body: Union[list, tuple]) -> str:
        """
        Create a defeasible rule.
        Args:
            -head: A literal (the head of the defeasible rule)
            -body: A list of literals (the body of the defeasible rule)
        """
        if isinstance(body[0], str):
            # Is fact
            body_string = 'true'
        else:
            # Is rule
            body_literals = self.get_body_literals(body)
            body_string = ','.join(body_literals)

        return str(head + ' ' + self.drule_symbol + ' ' + body_string + '.')
    

    def get_body_literals(self, body: list) -> list:
        """
        Get all literals in a body
        Args:
            -body: A list of tuples, each tuple is a rule in the body argument
        Output:
            - A list with all literals in body 
        """
        body_literals = []

        for tup_info in body:
            literal = self.levels[tup_info[1]][tup_info[0]][tup_info[2]][0]
            body_literals.append(literal)
        return body_literals


    def get_new_literal(self) -> str:
        """
        Create and return a new literal for the program
        Args:--
        Output:
            - A string that represent a literal 
        """
        polarity = self.utils.get_random()

        atom = 'a_' + self.get_new_id()
        if polarity < self.params["NEG_PROB"]:
            literal = self.utils.get_complement(atom)
        else:
            literal = atom
        
        return literal


    def add_to_kb(self, level: int, head: str, body: list, type: str) -> None:
        """
        Add a new argument to the KB
        Args:
            -level: The level to which the argument belongs
            -head: The head of the argument
            -body: Body of the argument
            -type: The type of rule with <head> as its consequent. Options:
                --'rnd': Randomly assign whether it will be a strict rule or 
                a defeasible rule (considering the params DEF_PROB).
                --'srule': Save as strict rule
                --'drule': Save as defeasible rule
        """
        if type == 'rnd':
            random_DS = self.utils.get_random()
            if random_DS < self.params["DRULE_PROB"]:
                self.levels[level]['drules'].append((head, body))
                self.rules['drules'].append(head)
            else:
                self.levels[level]['srules'].append((head, body))
                self.rules['srules'].append(head)
        else:
            self.levels[level][type].append((head, body))
            self.rules[type].append(head)



    def get_one_rule_level(self, level: int) -> tuple:
        """
        Select a rule from possibles rules to build an argument body
        Args:
            -level: A KB level
        Out:
            -tuple: A tuple of the form (type, level, pos)
                --type: 'drule' or 'srule'
                --level: Level of the rule
                --pos: Index of the rule in the level <level>
        """
        #self.utils.print_info(str(self.USED_HEADS))
        possibles_drules = [index for index, drule in 
                                    enumerate(self.levels[level]["drules"]) if 
                                    drule[0] not in self.USED_HEADS]
        possibles_srules = [index for index, srule in 
                                    enumerate(self.levels[level]["srules"]) if 
                                    srule[0] not in self.USED_HEADS]   
        random_DS = self.utils.get_random()
        if random_DS < self.params["DRULE_PROB"]:
            if len(possibles_drules) != 0:
                # Take a drule (its position) from level <level>
                index_drule = self.utils.get_choice(possibles_drules)
                rule = ('drules', level, index_drule)
            else:
                # Build a drule and put into the level <level>?
                #self.utils.print_error("No more drules!")
                lit = self.get_new_literal()
                self.levels[0]['drules'].append((lit, ('true',)))
                rule = ('drules', 0, len(self.levels[0]['drules']) - 1)
        else:
            if len(possibles_srules) != 0:
                # Take a srule (its position) from level <level>
                index_srule = self.utils.get_choice(possibles_srules)
                rule = ('srules', level, index_srule)
            else:
                # Build a srule and put into the level <level>?
                #self.utils.print_error("No more srules!")
                lit = self.get_new_literal()
                self.levels[0]['srules'].append((lit, ('true',)))
                rule = ('srules', 0, len(self.levels[0]['srules']) - 1)
        #self.utils.print_error(str(rule))
        return rule


    def build_body(self, level: int, conclusion: str) -> list:
        """
        Build an argument body (a list of tuples) whit at least one rule from
        a particular KB level.
        Args:
            -level: A KB level
            -conclusion: The conclusion of the argument for which we are 
            creating a body
        """
        body = ()
        
        self.USED_HEADS = self.USED_HEADS + (conclusion,)
        self.USED_HEADS = self.USED_HEADS + (self.utils.get_complement(conclusion),)

        body_size = self.utils.get_randint(1, self.params["MAX_BODYSIZE"])
        rule = self.get_one_rule_level(level - 1)
        self.USED_HEADS = self.USED_HEADS + (self.levels[rule[1]][rule[0]][rule[2]][0],)
        body = body + (rule,)

        for aux in range(body_size - 1):
            select_level = self.utils.get_choice(level)
            new_rule = self.get_one_rule_level(select_level)
            self.USED_HEADS = self.USED_HEADS + (self.levels[new_rule[1]][new_rule[0]][new_rule[2]][0],)
            body = body + (new_rule,) 
        
        self.USED_HEADS = ()
        return body 
    

    def clean_level(self, level: int) -> None:
        """
        Delte all duplicate in level <level>
        Args:
            -level: The level to clean
        """
        self.levels[level]['drules'] = list(set(self.levels[level]['drules']))
        self.levels[level]['srules'] = list(set(self.levels[level]['srules']))


    def build_arguments(self, level: int) -> None:
        """"
        Build all arguments for a particular level
        Args:
            -level: A KB level
        """
        # To save all arguments in this level
        self.levels[level] = {'drules': [], 'srules': []}
        # Min number of different argumens in the level
        min_args = self.params["MIN_ARGSLEVEL"]

        for i_aux in range(min_args):
            # Generate a new head (conclusion)
            head = self.get_new_literal()
            # To define how many arguments with the same head to create
            args_head = self.utils.get_randint(1, self.params["MAX_RULESPERHEAD"])
            # Build all arguments for <head>
            for j_aux in range(args_head):
                # Generete the body of the argument
                body = self.build_body(level, head)
                # Add to the KB
                self.add_to_kb(level, head, body, 'rnd') 
        self.clean_level(level)
   
    def build_complete_arguments(self, argument: list, tipo: str):
        """
        Build the complete argument
        Args:
            -argument: The argument as list of tuples (head, (body))
            -tipo:
                --'srule': The rule that derive the conclusion is strict
                --'drule': The rule that derive the conclusion is defeasible
        """
        if not isinstance(argument[1][0], str):
            body_literals = self.get_body_literals(argument[1])
            rule = [(argument[0], tipo, body_literals)]

            for in_body in argument[1]:
                arg = self.levels[in_body[1]][in_body[0]][in_body[2]]
                rule += self.build_complete_arguments(arg, in_body[0])
            return rule
        else:
            # Is fact or presumption
            return []  


    def build_actsets_ntactsets(self, complete_argument: list) -> dict:
        """
        Compute the Act-sets and NTAct-sets of the argument <complete_argument>
        Args:
            -complete_argument: The complete argument
            -conclusion: The conclusion of the argument
        Output:
            -dict:{
                'act_sets': set,
                'ntact_sets': set
            }
        """
        c = [[{conclusion,'a','b'}, 'trivial']]
        act_sets = {}
        ntact_sets = {}

        while True:
            [conj, tipo] = c.pop(0)
            for element in conj:
                rule = 'a' 
            if len(c) == 0:
                print("FIN")
                break
    
    def build_dialectical_trees(self) -> None:
        """
        To build all dialectical trees for each arguments
        """
        print(self.levels)
        for level, rules in self.levels.items():
            arguments = rules["drules"] + rules["srules"]
            for argument in arguments:
                argument = [rule for rule in argument if not isinstance(argument[1][0], str)]
                if len(argument) != 0:
                    complete_argument = self.build_complete_arguments(argument, 'drules')
                    #ntact_set = self.get_body_literals(arg[1])
                    #self.utils.print_ok("Head: " + arg[0] + "NTAct_set: " + str(ntact_set))


    def build_tree(self, literal: str, body: list, level: int, height: int) -> None:
        """
        Build the dialectical tree for a particular argument
        Args:
            literal: The conclusion of the root arguement
            body: The body of the root argument
            level: The KB level of the root argument
            height: The hight of the tree to be constructed
        """
        inners_point = [lit for lit in body if lit not in self.rules['srules']]
        if len(inners_point) != 0:
            # ramification is the max number of defeater for the actual argument
            # ramification = self.utils.get_randint(1, self.params.RAMIFICATION + 1)
            ramification = self.params.RAMIFICATION
            if height == 0:
                # Tree leaves
                for aux in range(ramification):
                    random_def_point = self.utils.get_random()
                    if random_def_point < self.params.INNER_PROB:
                        # Build a defeater for some litearl in the body
                        inner_point = self.utils.get_choice(inners_point)
                        complement = self.utils.get_complement(inner_point)
                        # defeater_body = build_body(level)
                        #defeater_body = self.build_body_def(body, complement, literal)
                        defeater_body = self.build_body_incremental(body)
                        self.KB[level]['drules'].append([complement, defeater_body])
                        self.rules['drules'].append(complement)
                        self.utils.print_info('****LEAF-inner*****')
                        self.utils.print_ok('Defeated Argument: [' + literal + ',' + str(body) + ']')
                        self.utils.print_ok('By: [' + complement + ',' + str(defeater_body) + ']')
                        self.utils.print_info('****LEAF*****')
                    else:
                        # Build a defeater for literal
                        complement = self.utils.get_complement(literal)
                        # defeater_body = build_body(level)
                        #defeater_body = self.build_body_def(body, complement, literal)
                        defeater_body = self.build_body_incremental(body)
                        self.KB[level]['drules'].append([complement, defeater_body])
                        self.rules['drules'].append(complement)
                        self.utils.print_info('****LEAF-conclusion*****')
                        self.utils.print_ok('Defeated Argument: [' + literal + ',' + str(body) + ']')
                        self.utils.print_ok('By: [' + complement + ',' + str(defeater_body) + ']')
                        self.utils.print_info('****LEAF*****')
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
                        
                        #defeater_body = self.build_body_def(body, complement, literal)
                        defeater_body = self.build_body_incremental(body)
                        self.KB[level]['drules'].append([complement, defeater_body])
                        self.rules['drules'].append(complement)
                        defeaters.append([complement, defeater_body])
                        self.utils.print_info('****INTERNAL-inner*****')
                        self.utils.print_ok('Defeated Argumenttt: [' + literal + ',' + str(body) + ']')
                        self.utils.print_ok('By: [' + complement + ',' + str(defeater_body) + ']')
                        self.utils.print_info('*********')

                    else:
                        # Build a defeater for literal
                        # and append in to defeaters list
                        complement = self.utils.get_complement(literal)
                        # defeater_body = build_body(level)
                        #defeater_body = self.build_body_def(body, complement, literal)
                        defeater_body = self.build_body_incremental(body)
                        self.KB[level]['drules'].append([complement, defeater_body])
                        self.rules['drules'].append(complement)
                        defeaters.append([complement, defeater_body])
                        self.utils.print_info('****INTERNAL-conclusion*****')
                        self.utils.print_ok('Defeated Argument: [' + literal + ',' + str(body) + ']')
                        self.utils.print_ok('By: [' + complement + ',' + str(defeater_body) + ']')
                        self.utils.print_info('*********')
                for defeater in defeaters:
                    self.build_tree(defeater[0], defeater[1], level, height - 1)
        else:
            pass


    def build_kb_base(self) -> None:
        """
        Build the KB Base (facts and presumptions only)
        """
        self.levels[0] = {'drules': [], 'srules': []}
        for i in range(self.params["KBBASE_SIZE"]):
            random_FP = self.utils.get_random()
            literal = self.get_new_literal()
            if random_FP < self.params["FACT_PROB"]:
                # New Fact
                self.levels[0]['srules'].append((literal, ('true',)))
            else:
                # New Presumption
                self.levels[0]['drules'].append((literal, ('true',)))


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
        to_string = 'use_criterion(' + self.params["PREF_CRITERION"] +').'
        program.append('')
        for key, value in self.levels.items():
            kb_level = str(key)
            program.append('/*** KB LEVEL = ' + kb_level + ' ***/')
            for drule in value['drules']:
                rule = self.create_def_rule(drule[0], drule[1])
                if rule not in program:
                    program.append(rule)
            for srule in value['srules']:
                rule = self.create_strict_rule(srule[0], srule[1])
                if rule not in program:
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
        
        for id_program in range(self.params["N_PROGRAMS"]):
            self.clear_datastructures()
            self.build_kb_base()
            self.build_kb(self.params["LEVELS"])
            self.to_delp_format(result_path, id_program)
            self.build_dialectical_trees() 
            #self.build_actsets_ntactsets([1,2,3], 'c')
             
