import math
import os
import time
from data import Dataset

from pysat.examples.rc2 import RC2
from pysat.formula import CNF
from pysat.solvers import Solver, Glucose3
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Learning Decision Trees with SAT')
    parser.add_argument('--data_file', help='Path to the dataset', required=True)
    parser.add_argument('--train_ratio', help='Training data ratio', type=float, default=0.75)
    parser.add_argument('--sat', help='SAT solver', default='g3')
    parser.add_argument('--size', help='Size of the tree', type=int, required=True)
    parser.add_argument('--n_learners', help='Number of learners', type=int, required=True)
    parser.add_argument('--seed', help='Random seed', type=int, default=2024)
    parser.add_argument('--separator', help='Separator for the dataset', default=' ')
    return parser.parse_args()


class DecisionTreeLearner:

    def __init__(self, data, options):
        self.data = data
        self.options = options
        self.cnf = CNF()
        self.solver = Solver(name=options.sat)

        # variables
        self.vpool = IDPool()
        self.left_child_vars = {} # l_ij
        self.right_child_vars = {} # r_ij
        self.parent_vars = {} # p_ji
        self.discrimination_vars = {} # a_rj
        self.activation_vars = {} # u_rj
        self.label_vars = {} # y_s
        self.d0_vars = {} # d0_rj
        self.d1_vars = {} # d1_rj
        self.class_vars = {} # c_j
        self.leaf_vars = {} # v_i
        self.depth_vars = {} # h_it
        self.size_vars = {} # m_i
        self.lambda_vars = {} # lambda_it
        self.tau_vars = {} # tau_it
        self.aux_vars = {} # aux_k

        # clauses
        self.clauses = []
        

    # ===================== Utility functions for variable generation =====================

    def get_l_bounds(self, i, N=None):
        return range(i + 1, min(2 * i, N - 1) + 1)

    def get_r_bounds(self, i, N=None):
        return range(i + 2, min((2 * i) + 1, N) + 1)

    def get_p_bounds(self, j, N=None):
        return range(max(j // 2, 1), min((j - 1), N) + 1)

    def get_lambda_bounds(self, i):
        return range(int(math.floor(i / 2)) + 1)

    def get_tau_bounds(self, i):
        return range(i + 1)

    # get the bound of the height for the node i
    def get_h_bounds(self, i):
        return range(int(math.ceil(np.log2(i+1))) - 1, int(math.ceil((i-1)*1.0/2)) + 1)
    

    def print_all_variables(self):
        print("Left child variables:")
        for i, left_children in self.left_child_vars.items():
            print(f"Node {i}: {left_children}")
        
        print("\nRight child variables:")
        for i, right_children in self.right_child_vars.items():
            print(f"Node {i}: {right_children}")
        
        print("\nParent variables:")
        for j, parents in self.parent_vars.items():
            print(f"Node {j}: {parents}")
        
        print("\nDiscrimination variables:")
        for r, discriminators in self.discrimination_vars.items():
            print(f"Feature {r}: {discriminators}")
        
        print("\nActivation variables:")
        for r, activators in self.activation_vars.items():
            print(f"Feature {r}: {activators}")
        
        print("\nD0 variables:")
        for r, d0s in self.d0_vars.items():
            print(f"Feature {r}: {d0s}")
        
        print("\nD1 variables:")
        for r, d1s in self.d1_vars.items():
            print(f"Feature {r}: {d1s}")
        
        print("\nClass variables:")
        for i, c_i in self.class_vars.items():
            print(f"Node {i}: {c_i}")
        
        print("\nLeaf node variables:")
        print(self.leaf_vars)
        
        print("\nAux variables:")
        print(self.aux_vars)

    # ===================== Variable generation functions

    def generate_left_child_variables(self, N):
        for i in range(1, N + 1):
            self.left_child_vars[i] = {}
            for j in self.get_l_bounds(i=i, N=N):
                if (j % 2 == 0):
                    self.left_child_vars[i][j] = self.vpool.id(f"l_{i}_{j}")


    def generate_right_child_variables(self, N):
        for i in range(1, N + 1):
            self.right_child_vars[i] = {}
            for j in self.get_r_bounds(i=i, N=N):
                if (j % 2 == 1):
                    self.right_child_vars[i][j] = self.vpool.id(f"r_{i}_{j}")


    def generate_parent_variables(self, N):
        for j in range(2, N + 1):
            self.parent_vars[j] = {}
            for i in self.get_p_bounds(j=j, N=N):
                self.parent_vars[j][i] = self.vpool.id(f"p_{j}_{i}")


    def generate_leaf_node_variables(self, N):
        for i in range(1, N + 1):
            self.leaf_vars[i] = self.vpool.id(f"v_{i}")


    def generate_lambda_variables(self, N):
        for i in range(1, N + 1):
            self.lambda_vars[i] = {}
            for t in self.get_lambda_bounds(i=i):
                self.lambda_vars[i][t] = self.vpool.id(f"lambda_{i}_{t}")


    def generate_tau_variables(self, N):
        for i in range(1, N + 1):
            self.tau_vars[i] = {}
            for t in self.get_tau_bounds(i=i):
                self.tau_vars[i][t] = self.vpool.id(f"tau_{i}_{t}")


    def generate_discrimination_variables(self, N, K):
        for r in range(1, K + 1):
            self.discrimination_vars[r] = {}
            for j in range(1, N + 1):
                self.discrimination_vars[r][j] = self.vpool.id(f"a_{r}_{j}")


    def generate_activation_variables(self, N, K):
        for r in range(1, K + 1):
            self.activation_vars[r] = {}
            for j in range(1, N + 1):
                self.activation_vars[r][j] = self.vpool.id(f"u_{r}_{j}")


    def generate_d0_variables(self, N, K):
        for r in range(1, K + 1):
            self.d0_vars[r] = {}
            for j in range(1, N + 1):
                self.d0_vars[r][j] = self.vpool.id(f"d0_{r}_{j}")


    def generate_d1_variables(self, N, K):
        for r in range(1, K + 1):
            self.d1_vars[r] = {}
            for j in range(1, N + 1):
                self.d1_vars[r][j] = self.vpool.id(f"d1_{r}_{j}")


    def generate_class_variables(self, N):
        for i in range(1, N + 1):
            self.class_vars[i] = self.vpool.id(f"c_{i}")


    def generate_variables(self, N, K, max_depth=-1, depth=-1):
        """
        Generate the variables of the MaxSAT instance using a variable pool.
        """

        self.generate_left_child_variables(N)
        self.generate_right_child_variables(N)
        self.generate_leaf_node_variables(N)
        self.generate_parent_variables(N)
        self.generate_discrimination_variables(N, K)
        self.generate_activation_variables(N, K)
        self.generate_d0_variables(N, K)
        self.generate_d1_variables(N, K)
        self.generate_class_variables(N)

        self.print_all_variables()


    # ===================== Constraint util functions

    def add_clause(self, clause):

        self.cnf.append(clause)
        self.clauses.append(clause)


    # ===================== Constraint generation functions

    def add_root_not_leaf_constraint(self):
        """
        Equation 1 original paper.
        Adds the constraint that the root node is not a leaf: ¬v1
        """
        new_clause = [-self.leaf_vars[1]]
        self.add_clause(new_clause) # ¬v1


    def add_leaf_no_children_constraint(self):
        """
        Equation 2 original paper.
        Adds constraints that if a node is a leaf, it has no children.
        """
        for i, v_i in self.leaf_vars.items():
            for j in self.left_child_vars.get(i, {}):
                new_clause = [-v_i, -self.left_child_vars[i][j]]
                self.add_clause(new_clause)  # v_i -> ¬l_ij

    
    def add_consecutive_child_constraint(self):
        """
        Equation 3 original paper.
        Ensure left child of node i has a consecutive right child.
        """
        for i, left_children in self.left_child_vars.items():
            for j, l_ij in left_children.items():
                r_ij_plus_1 = self.right_child_vars.get(i, {}).get(j + 1)
                if r_ij_plus_1:
                    new_clause = [l_ij, -r_ij_plus_1]
                    self.add_clause(new_clause)  # l_ij <- r_ij+1
                    new_clause = [-l_ij, r_ij_plus_1]
                    self.add_clause(new_clause)  # l_ij -> r_ij+1
                else:
                    raise ValueError(f"Right child not found for left child {l_ij} of node {i}")


    def add_non_leaf_must_have_child_constraint(self):
        """
        Equation 4 original paper.
        Ensure non-leaf nodes have exactly one left child (implicitely also the right ones, 
        because of equation 3).
        """
        for i, v_i in self.leaf_vars.items():
            children = list(self.left_child_vars.get(i, {}).values())
            print("children:", children)
            if children:
                card = CardEnc.equals(lits=children, bound=1, vpool=self.vpool,encoding=EncType.pairwise)
                print("card clauses:", card.clauses)
                for new_clause in card.clauses:
                    self.add_clause([v_i] + new_clause)

                    # Add the new variables created for the encoding to the vpool
                    for literal in new_clause:
                        var = abs(literal)
                        if not self.vpool.obj(var):
                            self.aux_vars[len(self.aux_vars)] = self.vpool.id(f"aux_{var}_eq4")


    def add_parent_child_relationship(self):
        """
        Equation 5 original paper.
        Ensure parent-child relationship through left or right indicators.
        """
        for j, parents in self.parent_vars.items():
            for i, p_ji in parents.items():
                if j in self.left_child_vars.get(i, {}):
                    l_ij = self.left_child_vars[i][j]
                    new_clause = [p_ji, -l_ij]
                    self.add_clause(new_clause)
                    new_clause = [-p_ji, l_ij]
                    self.add_clause(new_clause)
                    # print(f"Left child found for parent p_{j}_{i}")
                # else:
                    # print(f"Left child not found for parent p_{j}_{i}")
                if j in self.right_child_vars.get(i, {}):
                    r_ij = self.right_child_vars[i][j]
                    new_clause = [p_ji, -r_ij]
                    self.add_clause(new_clause)
                    new_clause = [-p_ji, r_ij]
                    self.add_clause(new_clause)
                    # print(f"Right child found for parent p_{j}_{i}")
                # else:
                    # print(f"Right child not found for parent p_{j}_{i}")


    def add_tree_structure_constraint(self, N):
        """
        Equation 6 original paper.
        Ensure each non-root node has exactly one parent.

        Args:
            N (int): Number of nodes
        """
        for j in range(2, N + 1):
            parents = list(self.parent_vars[j].values())
            card = CardEnc.equals(lits=parents, bound=1, vpool=self.vpool,encoding=EncType.pairwise)
            for new_clause in card.clauses:
                self.add_clause(new_clause)

                # Add the new variables created for the encoding to the vpool
                for literal in new_clause:
                    var = abs(literal)
                    if not self.vpool.obj(var):
                        self.aux_vars[len(self.aux_vars)] = self.vpool.id(f"aux_{var}_eq6")


    def recursively_convert_to_cnf(self, terms, orig_terms, aux_term, k, p):

        new_clause = aux_term.copy() + [orig_terms[k][p]]

        if k == len(orig_terms) - 1:
            print(k, p)
            terms.append(new_clause)
            return

        # we assume that the number of literals is the same for each clause
        for q in range(len(orig_terms[k])):
            self.recursively_convert_to_cnf(terms, orig_terms, new_clause, k + 1, q)
            


    def add_discrimination_for_value_0(self, N):
        """
        Equation 7 original paper.
        Define discrimination constraints for feature value 0.

        Args:
            N (int): Number of nodes.
        """
        for r, d0_r in self.d0_vars.items():
            new_clause = [-d0_r[1]]
            self.add_clause(new_clause)
            for j in range(2, N + 1):
                terms = []
                for i in range(j // 2, j):
                    if i in self.parent_vars.get(j, {}) and i in self.d0_vars.get(r, {}) and j in self.right_child_vars.get(i, {}):
                        p_ji = self.parent_vars[j][i]
                        d0_ri = self.d0_vars[r][i]
                        a_ri = self.discrimination_vars[r][i]
                        r_ij = self.right_child_vars[i][j]
                        terms.append([p_ji, d0_ri])
                        terms.append([a_ri, r_ij])
                if terms:
                    # First implication (already in CNF)
                    for term in terms:
                        new_clause = [d0_r[j]] + [-lit for lit in term]
                        self.add_clause(new_clause)
                        
                    # Distribute the -d0_rj to the terms to convert to CNF
                    print("Original terms:")
                    self.print_all_constraints(terms)
                    cnf_terms = []
                    self.recursively_convert_to_cnf(cnf_terms, terms, [-d0_r[j]], 0, 0)
                    print("CNF terms:")
                    self.print_all_constraints(cnf_terms)

                    # Convert the DNF formula to a CNF formula, pass the top variable of vpool
                    # terms_CNF = CNF(from_clauses=terms)
                    # terms_CNF = terms_CNF.negate(topv=self.vpool.top)
                    # for term in terms_CNF.clauses:
                    #     new_clause = [-d0_r[j]] + term
                    #     self.add_clause(new_clause)

                    #     # Add the new variables created for the encoding to the vpool
                    #     for literal in term:
                    #         var = abs(literal)
                    #         if not self.vpool.obj(var):
                    #             self.aux_vars[len(self.aux_vars)] = self.vpool.id(f"aux_{var}_eq7")


    def add_discrimination_for_value_1(self, N):
        """
        Equation 8 original paper.
        Define discrimination constraints for feature value 1.

        Args:
            N (int): Number of nodes.
        """
        for r, d1_r in self.d1_vars.items():
            new_clause = [-d1_r[1]]
            self.add_clause(new_clause)
            for j in range(2, N + 1):
                neg_terms = []
                for i in range(j // 2, j):
                    if i in self.parent_vars.get(j, {}) and i in self.d1_vars.get(r, {}) and j in self.left_child_vars.get(i, {}):
                        p_ji = self.parent_vars[j][i]
                        d1_ri = self.d1_vars[r][i]
                        a_ri = self.discrimination_vars[r][i]
                        l_ij = self.left_child_vars[i][j]
                        neg_terms.append([-p_ji, -d1_ri])
                        neg_terms.append([-a_ri, -l_ij])
                if neg_terms:
                    # Tricks for getting both the CNF formula and its negation, also in CNF
                    for neg_term in neg_terms:
                        new_clause = [d1_r[j]] + neg_term
                        self.add_clause(new_clause)

                    # Convert the -terms CNF formula to another CNF formula, pass the top variable of vpool
                    neg_terms_CNF = CNF(from_clauses=neg_terms)
                    terms_CNF = neg_terms_CNF.negate(topv=self.vpool.top)
                    for term in terms_CNF.clauses:
                        new_clause = [-d1_r[j]] + term
                        self.add_clause(new_clause)

                        # Add the new variables created for the encoding to the vpool
                        for literal in term:
                            var = abs(literal)
                            if not self.vpool.obj(var):
                                self.aux_vars[len(self.aux_vars)] = self.vpool.id(f"aux_{var}_eq8")


    def add_path_activation_constraint(self, N, K):
        """
        Implements Equation (9) of original paper to enforce path activation constraints for features.
        Ensures that if a feature `r` is used at node `j`, then its activation along the path is consistent.

        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        for r in range(1, K + 1):  
            for j in range(2, N + 1):  
                
                for i in range(j // 2, j):
                    if i in self.parent_vars.get(j, {}):
                        u_ri = self.activation_vars[r].get(i)
                        p_ji = self.parent_vars[j].get(i)
                        a_rj = self.discrimination_vars[r].get(j)
                        assert u_ri is not None and p_ji is not None and a_rj is not None
                        self.add_clause([-u_ri, -p_ji, -a_rj]) # ¬u_ri ∨ ¬p_ji ∨ ¬a_rj
                    else:
                        raise ValueError(f"Parent not found for node {j}")
                
                u_rj = self.activation_vars[r].get(j)
                a_rj = self.discrimination_vars[r].get(j)
                neg_terms = []
                for i in range(j // 2, j):
                    u_ri = self.activation_vars[r].get(i)
                    p_ji = self.parent_vars[j].get(i)
                    assert u_ri is not None and p_ji is not None
                    neg_terms.append([-u_ri, -p_ji])

                assert u_rj is not None and a_rj is not None
                if not neg_terms:
                    raise ValueError("No negated terms found for path activation constraint")

                # Clause: (a_rj ∨ ∨ (u_ri ∧ p_ji)) → u_rj
                self.add_clause([-a_rj, u_rj])

                for neg_term in neg_terms:
                    self.add_clause(neg_term + [u_rj])
                
                # Clause: u_rj → (a_rj ∨ ∨ (u_ri ∧ p_ji))
                neg_terms_CNF = CNF(from_clauses=neg_terms)
                terms_CNF = neg_terms_CNF.negate(topv=self.vpool.top)
                for term in terms_CNF.clauses:
                    new_clause = [-u_rj] + term
                    self.add_clause(new_clause)

                    # Add the new variables created for the encoding to the vpool
                    for literal in term:
                        var = abs(literal)
                        if not self.vpool.obj(var):
                            self.aux_vars[len(self.aux_vars)] = self.vpool.id(f"aux_{var}_eq9")


    def add_feature_usage_constraints(self, N, K):
        """
        Enforces that:
        1. Non-leaf nodes use exactly one feature.
        2. Leaf nodes use no features.
        
        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        for j in range(1, N + 1):
            v_j = self.leaf_vars[j]
            
            feature_usage = [self.discrimination_vars[r][j] for r in range(1, K + 1)]
            
            # Equation (10) of original paper: Exactly one feature is used for non-leaf nodes
            if feature_usage:
                card = CardEnc.equals(lits=feature_usage, bound=1, vpool=self.vpool,encoding=EncType.pairwise)
                for clause in card.clauses:
                    self.add_clause([v_j] + clause)

                    # Add the new variables created for the encoding to the vpool
                    for literal in clause:
                        var = abs(literal)
                        if not self.vpool.obj(var):
                            self.aux_vars[len(self.aux_vars)] = self.vpool.id(f"aux_{var}_eq10")

                # Equation (11) of original paper: No feature is used for leaf nodes
                for a_rj in feature_usage:
                    # If node `j` is a leaf (v_j), then no feature should be active (¬a_rj for each feature `r`)
                    self.add_clause([-v_j, -a_rj])  # v_j -> ¬a_rj
            else:
                raise ValueError("No feature usage found for node", j)


    def add_positive_leaf_discriminative_feature_constraint(self, N, K):
        """
        Enforces that if a leaf node `j` is assigned a positive class,
        then at least one discrimination variable must be active for that node
        based on the sign of the feature in the example.
        
        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        positive_samples = self.data.get_positive_train_samples()
        for q, eq in positive_samples:
            
            for i in range(1, N + 1):
                v_i = self.leaf_vars[i]
                c_i = self.class_vars.get(i)
                
                discriminative_features = []
                for r in range(1, K + 1):
                    feature_value = eq[r - 1]
                    d_ri = self.d0_vars[r][i] if feature_value == 0 else self.d1_vars[r][i]
                    discriminative_features.append(d_ri)
                
                # Add clause for positive example: ¬v_i ∨ c_i ∨ (∨ d_{sigma(r, q)})
                clause = [-v_i, c_i] + discriminative_features
                self.add_clause(clause)


    def add_negative_leaf_discriminative_feature_constraint(self, N, K):
        """
        Enforces that if a leaf node `j` is assigned a negative class,
        then at least one discrimination variable must be active for that node
        based on the sign of the feature in the example.

        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        negative_samples = self.data.get_negative_train_samples()
        for q, eq in negative_samples:
            
            for i in range(1, N + 1):
                v_i = self.leaf_vars[i]
                c_i = self.class_vars.get(i)
                
                discriminative_features = []
                for r in range(1, K + 1):
                    feature_value = eq[r - 1]
                    d_ri = self.d0_vars[r][i] if feature_value == 0 else self.d1_vars[r][i]
                    discriminative_features.append(d_ri)
                
                # Add clause for negative example: ¬b_q ∨ ¬v_i ∨ ¬c_i ∨ (∨ d_{sigma(r, q)})
                clause = [-v_i, -c_i] + discriminative_features
                self.add_clause(clause)


    def print_all_constraints(self, clauses):

        print("Clauses:")
        for clause in clauses:
            # map back the variable ids to their names
            for lit in clause:
                sign = str(np.sign(lit))[0] if lit < 0 else ""
                var = self.vpool.obj(abs(lit))
                print(f"{sign}{var}", end=" ")
            print()


    def generate_constraints(self, N, K):
        self.add_root_not_leaf_constraint()
        self.add_leaf_no_children_constraint()
        self.add_consecutive_child_constraint()
        self.add_non_leaf_must_have_child_constraint()
        self.add_parent_child_relationship()
        self.add_tree_structure_constraint(N)
        # self.add_discrimination_for_value_0(N)
        # self.add_discrimination_for_value_1(N)
        # self.add_path_activation_constraint(N, K)
        self.add_feature_usage_constraints(N, K)
        self.add_positive_leaf_discriminative_feature_constraint(N, K)
        self.add_negative_leaf_discriminative_feature_constraint(N, K)

        self.print_all_constraints(self.cnf.clauses)


    # ===================== Generate Decision Tree

    def generate_decision_tree(self, N, K=None, sol_path=None, u_wght_soft=[]):
        """
        Generate a decision tree using MaxSAT via RC2 solver.

        Args:
            N (int): Number of nodes.
            K (int, optional): Number of features. If None, inferred from data.
            max_depth (int): Maximum tree depth, mutually exclusive with `depth`.
            depth (int): Exact tree depth, mutually exclusive with `max_depth`.
            sol_path (str): Path for saving the solution.
            u_wght_soft (list): List of weights for soft clauses.

        Returns:
            Tuple with cost, solution path, classification results, and formula stats.
        """
        if K is None:
            K = len(self.data.train_samples[0]) - 1

        # Create or update the CNF formula
        self.generate_variables(N=N, K=K)
        self.generate_constraints(N=N, K=K)

        # Set up the base path for the CNF file
        file_name = os.path.splitext(os.path.basename(self.options.data_file))[0]
        cnf_base_path = self.create_dir_solution('binarytree/cnf', file_name)

        # Define the CNF file path based on depth or max_depth settings
        self.cnf_file_path = os.path.join(cnf_base_path, f"formula_{self.options.seed}_{N}.cnf")
        
        # Write the CNF formula to file
        self.cnf.to_file(self.cnf_file_path)

        # Log CNF formula stats
        n_var, n_clauses = self.cnf.nv, len(self.cnf.clauses)
        n_literals = sum(len(cl) for cl in self.cnf.clauses)
        print(f"n variables: {n_var}, n hard clauses: {n_clauses}, n literals: {n_literals}")

        print("Solving the CNF formula...")
        print(self.cnf.clauses)

        # Solve the CNF formula with the solver defined in self.solver
        with Glucose3(bootstrap_with=self.cnf.clauses) as g:
            g.solve()
            var_model = g.get_model()
            print("Model:", var_model)

        # Check the solution
        self.check_solution(var_model)

        # Clean up CNF file to save disk space
        if os.path.isfile(self.cnf_file_path):
            print(f"Deleting CNF file at {self.cnf_file_path} to save space.")
            os.remove(self.cnf_file_path)

        # Prepare solution file
        sol_file_name = sol_path if sol_path else f"{cnf_base_path}/formula_{self.options.seed}_{N}_best.sol"
        if var_model:
            self.build_graph(N=N, model=var_model, filename=sol_file_name, K=K, labeled=True)

        return sol_file_name, n_var, n_clauses, n_literals

    

    def analye_model_polarity(self, var_model):
        positive_count = sum(1 for var in var_model if var > 0)
        negative_count = sum(1 for var in var_model if var < 0)
        print(f"Positive literals: {positive_count}, Negative literals: {negative_count}")


    def analyze_clause_polarity(self):
        positive_count = sum(1 for clause in self.cnf.hard + self.cnf.soft for lit in clause if lit > 0)
        negative_count = sum(1 for clause in self.cnf.hard + self.cnf.soft for lit in clause if lit < 0)
        print(f"Positive literals: {positive_count}, Negative literals: {negative_count}")


    def print_constraints(self):
        """Print all constraints (clauses) added to the solver."""
        print("Constraints (Clauses):")
        for clause in self.clauses:
            print(clause)


    def create_dir_solution(self, basepath, dataset_name):
        path = os.path.join(basepath, dataset_name)
        print("Saving the cnf file at: ", path)
        if not os.path.exists(path):
            os.makedirs(path)
        return path


    # ===================== Check solution

    def check_solution(self, var_model):
        """
        Check if all the constraints are satisfied by the solution, and print the constraints that are violated.

        Args:
            var_model (list of int): Solution model from the solver, with positive values indicating true literals.
        """

        print("Checking the solution...")
        for clause in self.cnf.clauses:
            satisfied = False
            for literal in clause:
                if literal in var_model:
                    satisfied = True
                    break
            # it will never enter here, cause if that happened, the solver would return UNSAT
            if not satisfied:
                orig_clause = [self.vpool.obj(abs(literal)) for literal in clause]
                print("Constraint violated:", orig_clause)
    

    def build_graph(self, N, model, filename="computed_binary_dt.txt", K=None, labeled=False):
        """
        Build the graph based on the model solution and save it to a text file.
        
        Args:
            N (int): Total number of nodes in the tree.
            model (list of int): Solution model from the solver.
            filename (str): Output file name to store the graph information.
            K (int): Number of features, if available.
            labeled (bool): Whether to label nodes based on features.
        """
        
        with open(filename, "w") as file_dt:
            nodes = []
            labeldict = None
            
            # Determine number of used nodes based on `atleast` option
            n_used_nodes = N
            
            print("Number of nodes used:", n_used_nodes)
            file_dt.write("NODES\n")

            if labeled:
                labeldict = {}

                # Label each node with the feature used for branching
                for r in range(1, K + 1):
                    for j in range(1, n_used_nodes + 1):
                        a_rj = self.discrimination_vars[r].get(j)
                        assert a_rj is not None
                        print(f"a_{r}_{j}:", a_rj, len(model))
                        if model[a_rj - 1] > 0:
                            print(f"a_{r}_{j}:", a_rj, model[a_rj - 1] if a_rj is not None else None)
                            labeldict[j] = f"{self.data.feature_names[r - 1]}_{r}"
                            nodes.append((j, self.data.feature_names[r - 1]))
                            file_dt.write(f"{j} {self.data.feature_names[r - 1]}\n")
                            print(f"Node {j} labeled with feature: {self.data.feature_names[r - 1]}")

                # Label class nodes based on leaf and class variables
                for j in range(1, n_used_nodes + 1):
                    v_j = self.leaf_vars.get(j)
                    c_j = self.class_vars.get(j)
                    if v_j is not None and model[v_j - 1] > 0:
                        v = 1 if (c_j is not None and model[c_j - 1] > 0) else 0
                        labeldict[j] = f"c_{v}"
                        file_dt.write(f"{j} c_{v}\n")
                        print(f"Node {j} labeled with class: c_{v}")

            file_dt.write("EDGES\n")
            edges = []

            # Add edges for left and right children based on the model
            for i in range(1, n_used_nodes + 1):
                # Left child edges
                for j in self.get_l_bounds(i=i, N=n_used_nodes):
                    if j % 2 == 0:
                        l_ij = self.left_child_vars.get(i, {}).get(j)
                        if l_ij is not None and model[l_ij - 1] > 0:
                            edges.append((i, j, 1))
                            file_dt.write(f"{i} {j} 1\n")
                            print(f"Edge from node {i} to left child {j}")

                # Right child edges
                for j in self.get_r_bounds(i=i, N=n_used_nodes):
                    if j % 2 == 1:
                        r_ij = self.right_child_vars.get(i, {}).get(j)
                        if r_ij is not None and model[r_ij - 1] > 0:
                            edges.append((i, j, 0))
                            file_dt.write(f"{i} {j} 0\n")
                            print(f"Edge from node {i} to right child {j}")


    def get_number_nodes_really_used(self, N, model):
        """
        Get the number of nodes really used based on the model solution.
        
        Args:
            N (int): Total number of nodes.
            model (list of int): Solution model returned by the solver, where positive
                                literals indicate satisfied variables.
        
        Returns:
            int: The highest index of nodes used.
        """
        assert N % 2 == 1
        n_used_nodes = 1
        
        for i in range(1, N + 1):
            if i % 2 == 0: continue
            
            # Retrieve the variable ID for `m_i` directly from `vpool`
            m_var_id = self.size_vars.get(i)
            print(m_var_id, model[m_var_id - 1], self.vpool.obj(m_var_id))
            
            # Check if this node is used in the model (positive in the solution)
            if m_var_id is None:
                print(f"Warning: Size variable m_{i} not found in the variable pool.")
            if m_var_id is not None and model[m_var_id - 1] > 0:
                n_used_nodes = i
            else:
                break
            
        return n_used_nodes



def main():
    
    args = parse_args()
    data = Dataset(file_path=args.data_file, train_ratio=args.train_ratio, seed=args.seed, separator=args.separator)

    # we get the number of features and samples from the data
    n_features = len(data.feature_names) - 1
    n_samples = len(data.train_samples)
    print(f"Loaded {n_samples} samples with {n_features} features.")
    print(data.get_negative_train_samples())
    print(data.get_positive_train_samples())

    # initialize a decision tree learner with the data
    learner = DecisionTreeLearner(data=data, options=args)
    print(learner.generate_decision_tree(N=args.size, K=n_features))
    




if __name__ == '__main__':
    # get execution time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))