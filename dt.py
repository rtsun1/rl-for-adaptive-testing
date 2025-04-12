import time
import os
import pickle
import numpy as np
import pandas as pd
from dataloader import dataset_rl as dataset_dt

ABILITY_MIN = -4
ABILITY_MAX = 4

NUM_QUESTIONS = 244



def discretize_ability_train(num_classes: int):
    '''
    Add a new column which has the discretized ability level
    '''

    train_set, _, _ = dataset_dt()

    bins = np.linspace(ABILITY_MIN, ABILITY_MAX, num_classes+1)
    labels = range(num_classes)

    train_set["ability_class"] = pd.cut(train_set["final_score"], bins=bins, labels=labels)

    return train_set



'''
Reference: https://pypi.org/project/c45-decision-tree/
    - I only borrowed the overall structure, not the implementation
    - I borrowed the idea of the look-up tables from my teammate Reuben
'''
class QuestionNode:
    def __init__(self, item_id, time_taken, time_remain, num_qs_so_far, ability, left, right, current_time_table, current_response_table):
        '''
        item_id: question id
        time_taken: how long does it take to answer the current question
        time_remain: how many time left
        num_qs_so_far: how many question selected so far
        ability: current ability estimate so far
        left: question if the current question is answered wrongly
        right: question if the current question is answered correctly
        current_time_table: columns: questions available, rows: students available, entry: time taken
        current_response_table: columns: questions available, rows: students available, entry: answered correctly or wrongly
        '''

        self.item_id = item_id
        self.time_taken = time_taken
        self.time_remain = time_remain
        self.num_qs_so_far = num_qs_so_far
        self.ability = ability

        self.current_response_table = current_response_table
        self.current_time_table = current_time_table

        self.left = left
        self.right = right



class DecisionTree:
    def __init__(self, model: str,
                 time_aware: bool, 
                 mean_or_median: str, 
                 num_classes: int,
                 max_time: int,
                 max_qs):
        '''
        model: model name
        time_aware: want to divide e.g. IG by T or no
        mean_or_median: when computing time taken, use mean or median over the students
        num_classes: how many classes to split (-4, 4) ie ability range into
        max_time: max time of the exam
        max_qs: max number of the questions in the exam
        '''

        train_set = discretize_ability_train(num_classes)

        # look-up tables
        self.time_table = train_set.pivot_table(index="student_id", columns="item_id", values="response_time")
        self.response_table = train_set.pivot_table(index="student_id", columns="item_id", values="response")
        self.ability_table = train_set[["student_id", "final_score", "ability_class"]].drop_duplicates().reset_index(drop=True).set_index("student_id")

        assert model in ["c45", "id3", "cart"]
        assert mean_or_median in ["mean", "median"]

        self.model = model
        self.time_aware = time_aware
        self.mean_or_median = mean_or_median

        self.max_time = max_time
        self.max_qs = max_qs

        self.root_q = None


    def __entropy(self, current_response_table):
        '''
        - p_i = |S_i| / |S|
        - entropy = \sum_i - p_i * log_2(p_i)
        '''

        current_ability_set = self.ability_table.loc[current_response_table.index]["ability_class"].values
        prob = np.bincount(current_ability_set) / len(current_ability_set)
        entropy = -np.sum(prob * np.log2(np.where(prob > 0, prob, 1)))
        
        return entropy
    

    def __gini(self, current_response_table):
        '''
        - p_i = |S_i| / |S|
        - gini = 1 - \sum_i pi^2
        '''
        
        current_ability_set = self.ability_table.loc[current_response_table.index]["ability_class"].values
        prob = np.bincount(current_ability_set) / len(current_ability_set)
        gini = 1 - np.sum(prob ** 2)

        return gini


    def __split_info(self, left_prob, right_prob):
        '''
        - p_i = |S_i| / |S|
        - split info = \sum_i p_i * log2(p_i)
        '''

        return - left_prob * np.log2(np.where(left_prob > 0, left_prob, 1)) - right_prob * np.log2(np.where(right_prob > 0, right_prob, 1))
    

    def __compute_IG(self, item_id, current_response_table):
        '''
        IG = current IG - expected IG conditioning on some further question
        '''

        if len(current_response_table) == 0:
            return -np.inf

        current_response_table_left = current_response_table[current_response_table[item_id] == 0]
        current_response_table_right = current_response_table[current_response_table[item_id] == 1]

        if len(current_response_table_left) == 0 or len(current_response_table_right) == 0:
            return -np.inf

        left_prob = len(current_response_table_left) / len(current_response_table)
        right_prob = len(current_response_table_right) / len(current_response_table)

        IG = self.__entropy(current_response_table) - left_prob * self.__entropy(current_response_table_left) - right_prob * self.__entropy(current_response_table_right)

        return IG


    def __compute_gain_ratio(self, item_id, current_response_table):
        '''
        gain ratio = IG / SI
        '''

        if len(current_response_table) == 0:
            return -np.inf

        current_response_table_left = current_response_table[current_response_table[item_id] == 0]
        current_response_table_right = current_response_table[current_response_table[item_id] == 1]

        if len(current_response_table_left) == 0 or len(current_response_table_right) == 0:
            return -np.inf

        left_prob = len(current_response_table_left) / len(current_response_table)
        right_prob = len(current_response_table_right) / len(current_response_table)

        IG = self.__entropy(current_response_table) - left_prob * self.__entropy(current_response_table_left) - right_prob * self.__entropy(current_response_table_right)
        SI = self.__split_info(left_prob, right_prob)

        return IG/SI
    

    def __compute_gini(self, item_id, current_response_table):
        '''
        Computes the expected gini conditioning on some further question
        '''

        if len(current_response_table) == 0:
            return np.inf

        current_response_table_left = current_response_table[current_response_table[item_id] == 0]
        current_response_table_right = current_response_table[current_response_table[item_id] == 1]

        if len(current_response_table_left) == 0 or len(current_response_table_right) == 0:
            return np.inf

        left_prob = len(current_response_table_left) / len(current_response_table)
        right_prob = len(current_response_table_right) / len(current_response_table)

        expected_gini = left_prob * self.__gini(current_response_table_left) + right_prob * self.__gini(current_response_table_right)

        return expected_gini
    

    def __compute_time(self, item_id, current_time_table):
        '''
        Compute the mean / median time over the students who answered the previous questions in a certain way
        '''

        return current_time_table[item_id].mean() if self.mean_or_median == "mean" else current_time_table[item_id].median()


    def __select_best_q(self, current_response_table, current_time_table):
        '''
        Select the best question given the previous questions and responses
        '''

        if current_response_table.empty:
            return None, None

        qs_available = current_response_table.columns

        best_idx = None
        if self.model == "c45":
            if not self.time_aware:
                qs_IG = np.array([self.__compute_gain_ratio(item_id, current_response_table) for item_id in qs_available])
                best_idx = np.argmax(qs_IG)
            else:
                qs_IG_T = np.array([self.__compute_gain_ratio(item_id, current_response_table) / self.__compute_time(item_id, current_time_table) for item_id in qs_available])
                best_idx = np.argmax(qs_IG_T)
        
        elif self.model == "id3":
            if not self.time_aware:
                IG = np.array([self.__compute_IG(item_id, current_response_table) for item_id in qs_available])
                best_idx = np.argmax(IG)
            else:
                IG_T = np.array([self.__compute_IG(item_id, current_response_table) / self.__compute_time(item_id, current_time_table) for item_id in qs_available])
                best_idx = np.argmax(IG_T)

        elif self.model == "cart":
            if not self.time_aware:
                qs_gini = np.array([self.__compute_gini(item_id, current_response_table) for item_id in qs_available])
                best_idx = np.argmin(qs_gini)
            else:
                qs_gini = np.array([self.__compute_gini(item_id, current_response_table) / self.__compute_time(item_id, current_time_table) for item_id in qs_available])
                best_idx = np.argmin(qs_gini)       # or maximize gini gain

        best_q_id = qs_available[best_idx]
        best_q_response_time = current_time_table[best_q_id].mean() if self.mean_or_median == "mean" else current_time_table[best_q_id].median()

        return best_q_id, best_q_response_time
    

    def __compute_ability(self, current_response_table):
        '''
        Compute the average of the remaining (conditional) student's ability scores
        '''

        current_final_score_col = self.ability_table.loc[current_response_table.index]["final_score"]
        return current_final_score_col.mean() if self.mean_or_median == "mean" else current_final_score_col.median()


    def __build_dt(self, root_q):
        '''
        Build tree rooted at root_q
        '''

        if root_q is None:
            return
        if root_q.num_qs_so_far >= self.max_qs or root_q.time_remain <= 0:
            return
        if root_q.current_response_table.empty:
            return

        # left question

        # remove students whose answer to the question isn't 0
        response_mask_left = (root_q.current_response_table[root_q.item_id] == 0)
        current_response_table_left = root_q.current_response_table[response_mask_left]
        current_time_table_left = root_q.current_time_table[response_mask_left]

        # remove the selected question
        if not current_response_table_left.empty:
            current_response_table_left = current_response_table_left.drop(columns=[root_q.item_id])
            current_time_table_left = current_time_table_left.drop(columns=[root_q.item_id])

        # remove questions that takes longer than the time remaining
        if not current_response_table_left.empty:
            column_repr_left = current_time_table_left.mean(axis=0) if self.mean_or_median == "mean" else current_time_table_left.median(axis=0)
            columns_to_keep_left = (column_repr_left < root_q.time_remain).index

            current_time_table_left = current_time_table_left.loc[:, columns_to_keep_left]
            current_response_table_left = current_response_table_left.loc[:, columns_to_keep_left]

        # select question
        left_q = None
        if not current_response_table_left.empty:
            left_item_id, left_time_taken = self.__select_best_q(current_response_table=current_response_table_left,
                                                                current_time_table=current_time_table_left,)
        
            if left_item_id is not None:
                time_remain_left = root_q.time_remain-left_time_taken

                left_q = QuestionNode(item_id=left_item_id,
                                    time_taken=left_time_taken,
                                    time_remain=time_remain_left,
                                    num_qs_so_far=root_q.num_qs_so_far+1,
                                    ability=self.__compute_ability(current_response_table_left),
                                    current_response_table=current_response_table_left,
                                    current_time_table=current_time_table_left,
                                    left=None,
                                    right=None)
        root_q.left = left_q

        # right question

        # remove students whose answer to the question isn't 1
        response_mask_right = (root_q.current_response_table[root_q.item_id] == 1)
        current_response_table_right = root_q.current_response_table[response_mask_right]
        current_time_table_right = root_q.current_time_table[response_mask_right]

        # remove the selected question
        if not current_response_table_right.empty:
            current_response_table_right = current_response_table_right.drop(columns=[root_q.item_id])
            current_time_table_right = current_time_table_right.drop(columns=[root_q.item_id])

        # remove questions that takes longer than the time remaining
        if not current_response_table_right.empty:
            column_repr_right = current_time_table_right.mean(axis=0) if self.mean_or_median == "mean" else current_time_table_right.median(axis=0)
            columns_to_keep_right = (column_repr_right < root_q.time_remain).index

            current_time_table_right = current_time_table_right.loc[:, columns_to_keep_right]
            current_response_table_right = current_response_table_right.loc[:, columns_to_keep_right]

        # select question
        right_q = None
        if not current_response_table_right.empty:
            right_item_id, right_time_taken = self.__select_best_q(current_response_table=current_response_table_right,
                                                                current_time_table=current_time_table_right,)
        
            if right_item_id is not None:
                time_remain_right = root_q.time_remain-right_time_taken

                right_q = QuestionNode(item_id=right_item_id,
                                    time_taken=right_time_taken,
                                    time_remain=time_remain_right,
                                    num_qs_so_far=root_q.num_qs_so_far+1,
                                    ability=self.__compute_ability(current_response_table_right),
                                    current_response_table=current_response_table_right,
                                    current_time_table=current_time_table_right,
                                    left=None,
                                    right=None)
        root_q.right = right_q

        self.__build_dt(root_q.left)
        self.__build_dt(root_q.right)


    def train(self):
        '''
        Build the whole decision tree and clean up the unnecessary attributes
        '''

        # choose the first question
        item_id, time_taken = self.__select_best_q(current_response_table=self.response_table,
                                                   current_time_table=self.time_table)

        time_remain_root = self.max_time-time_taken

        self.root_q = QuestionNode(item_id=item_id,
                                   time_taken=time_taken,
                                    time_remain=time_remain_root,
                                    num_qs_so_far=1,
                                    ability=self.__compute_ability(self.response_table),
                                    current_response_table=self.response_table,
                                    current_time_table=self.time_table,
                                    left=None,
                                    right=None)

        self.__build_dt(self.root_q)

        # reduce memory
        self.__shrink_tree(self.root_q)
        self.time_table = None
        self.response_table = None
        self.ability_table = None


    def __shrink_tree(self, current_node):
        '''
        Remove unnecessary attributes (or else the file of the saved parameters will be huge)
        '''

        if current_node is not None:
            current_node.current_response_table = None
            current_node.current_time_table = None

            self.__shrink_tree(current_node.left)
            self.__shrink_tree(current_node.right)



if __name__ == "__main__":
    pass
