import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
from utils import Utils


class Generator:

    # Symbols
    srule_symbol = '<-'
    drule_symbol = '-<'
    
    #Utils
    utils = Utils()

    def __init__(self) -> None:
        """
        Constructor

        Define default values for all hiperparameters
        and data structures
        """
        # Probability to Facts elements in the KB BASE
        self.FACT_PROB = 0.5

        # Probability of negated literals in the KB BASE
        self.NEG_PROB = 0.5

        # Probability of Defeasible Rules in KB
        self.DEF_RULE_PROB = 0.5

        # Probability of attack a inner point of an argument
        self.INN_POINT_PROB = 0.5

        # Max size of body argument
        self.MAX_BODY_SIZE = 1

        # Min number of different arguments in each level
        self.MIN_DIF_ARG_LEVEL = 1

        # Max number of arguments in each level
        self.MAX_ARG_LEVEL = 1

        # Max rules defining the same literal 
        # (MAX_RULES_PER_HEAD <= MAX_ARGUMENTS_PER_LEVEL)
        self.MAX_RULES_PER_HEAD = 1

        # Ramification factor for each dialectical tree
        # (Max number of defeater for each argument)
        self.RAMIFICATION = 1

        # Max height of dialectical trees
        self.TREE_HEIGHT = 1

        # Levels of the KB
        self.LEVELS = 1 
        """"""

        """
        Global values
        """
        # List to control rules defining the same literal
        # [Literal] = Literal to define one rule
        self.LITERALS = []
        # Index of literals
        self.COUNT_LIT = 0
        # Number of program
        self.N_PROGRAM = []
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

    def create_strict_rule(self, head: str, body: list) -> str:
        """
        Create a strict rule.
        Args:
            -head: A literal (the head of the stric rule)
            -body: A list of literals (the body of the strict rule)
        """
        body_string = ','.join(body)
        return str(head + ' ' + srule_symbol + ' ' + body_string + '.')


    def create_def_rule(self, head: str, body: str) -> str:
        """
        Create a defeasible rule.
        Args:
            -head: A literal (the head of the defeasible rule)
            -body: A list of literals (the body of the defeasible rule)
        """
        body_string = ','.join(body)
        return str(head + ' ' + drule_symbol + ' ' + body_string + '.')


    def get_one_rule_level(self, level: int) -> str:
        """
        Select a head rule (literal) from a particular level
        Args:
            -level: A KB level
        """
        random = self.utils.get_random()
        if random < DEF_RULE_PROB and len(self.KB[level]['drules']) != 0:
            index = self.utils.get_choice(len(KB[level]['drules']), 1)[0]
            return KB[level]['drules'][index][0]
        elif len(KB[level]['srules']) != 0:
            index = self.utils.get_choice(len(KB[level]['srules']), 1)[0]
            return KB[level]['srules'][index][0]
        else:
            index = self.utils.get_choice(len(KB[level]['drules']), 1)[0]
            return KB[level]['drules'][index][0]


    def get_one_rule_all_levels(self, top_level: int) -> str:
        """
        Select a head rule (literal) from a level in [0, top_level]
        Args:
            -top_level: A KB level
        """
        select_from = np.random.randint(0, top_level, 1)[0]
        return get_one_rule_level(select_from)


    def build_body(level: int) -> list:
        """
        Build a list of literals whit at least one litera from
        a particular KB level.
        Args:
            -level: A KB level
        """
        body = []
        body_size = np.random.randint(1, MAX_BODY_SIZE + 1, 1)[0]
        body.append(get_one_rule_level(level - 1))

        for aux in range(body_size - 1):
            body.append(get_one_rule_all_levels(level))

        return body


    def build_body_def(body_dim: int) -> list:
        """
        PENDING!!!
        Build a list of literals with len body_dim
        Args:
            -body_dim: Size of the body (number of literals)
        """
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


    def build_arguments(level: int) -> None:
        """
        Build all arguments for a particular level
        Args:
            -level: A KB level
        """
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


    def build_tree(self, literal: str, body: list, level: int, height: int) -> None:
        """
        Build the dialectical tree for a particular argument
        Args:
            literal: The conclusion of the root arguement
            body: The body of the root argument
            level: The KB level of the root argument
            height: The hight of the tree to be constructed
        """
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
                        complement = self.utils.get_complement(inner_point)
                        # defeater_body = build_body(level)
                        defeater_body = build_body_def(len(body))
                        KB[level]['drules'].append([complement, defeater_body])
                        print("Lit: " + complement + " Body: " + str(defeater_body))
                    else:
                        # Build a defeater for literal
                        complement = self.utils.get_complement(literal)
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
                        complement = self.utils.get_complement(inner_point)
                        # defeater_body = build_body(level)
                        defeater_body = build_body_def(len(body))
                        KB[level]['drules'].append([complement, defeater_body])
                        defeaters.append([complement, defeater_body])
                        print("Lit: " + complement + " Body: " + str(defeater_body))
                    else:
                        # Build a defeater for literal
                        # and append in to defeaters list
                        complement = self.utils.get_complement(literal)
                        # defeater_body = build_body(level)
                        defeater_body = build_body_def(len(body))
                        KB[level]['drules'].append([complement, defeater_body])
                        defeaters.append([complement, defeater_body])
                        print("Lit: " + complement + " Body: " + str(defeater_body))
                for defeater in defeaters:
                    build_tree(defeater[0], defeater[1], level, height - 1)
        else:
            pass


    def build_kb_base(dim_kb_base: int) -> None:
        """
        Build the KB Base (facts and presumptions only)
        Args:
            -dim_kb_base: Size of the KB Base (total number of facts and
            presumptions)
        """
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


    def build_kb(level: int) -> None:
        """
        Build the KB (program) from level 1 to <level>
        Args:
            -level: Total number of KB levels
        """
        if level == 1:
            build_arguments(level)
        else:
            build_kb(level - 1)
            build_arguments(level)


    def to_delp_format(self, result_path: str) -> None:
        """
        Save the delp program
        Args:
            result_path: The path for save the program
        """
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


    def generate(self, result_path: str) -> None:
        """
        Generate a delp program
        Args:
            -result_path: The path to save the program
        """
        build_kb_base(10)
        build_kb(LEVELS)
        to_delp_format(result_path)
        # print(KB)
