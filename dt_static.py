import time
import os
import pickle
import numpy as np
import pandas as pd
from dataloader import dataset_rl as dataset_dt

ABILITY_MIN = -4
ABILITY_MAX = 4

MAX_QUESTIONS = 15
MAX_TIME = 900

NUM_QUESTIONS = 244


def discretize_ability_train(num_classes: int = 8):
    '''
    Add a new column which has the discretized ability level
    
    Parameters: 
        - num_classes (int): the number of classes to divide the ability range into

    Returns:
        - train_set, val_set, test_set (all pd dataframe) with one more column of the discretized ability level
    '''

    train_set, val_set, test_set = dataset_dt()

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
    def __init__(self, item_id, time_taken, time_remain, num_qs_so_far, left, right, qs_available, current_set):
        '''
        self.item_id (str): question name
        self.time_remain (float): Time remaining AFTER answering the current question (assuming that correct/wrong takes the same amount of time)
        self.num_qs_so_far (int): 0 ~ 15
        self.current_set (pd df): shrinked version of stu_answer_table that keeps track of the subset given the answered questions
        self.qs_available (pd df): with column "item_id"
        self.left (QuestionNode or None)
        self.right (QuestionNode or None)
        '''

        self.item_id = item_id
        self.time_taken = time_taken

        self.time_remain = time_remain
        self.num_qs_so_far = num_qs_so_far
        self.current_set = current_set
        self.qs_available = qs_available

        self.left = left
        self.right = right



class DTDiscreteStatic:
    def __init__(self, model: str, 
                 time_aware: bool, 
                 mean_or_median: str,
                 num_classes: int):
        '''
        self.root_q (QuestionNode)
        self.model (str): model name (e.g. c45, id3, cart)
        self.time_aware (bool): True if you want the criterion (e.g. IG) to be divided by avg time
        '''

        train_set = discretize_ability_train(num_classes)

        # look-up tables
        self.q_bank = train_set.groupby("item_id")["response_time"].agg(response_time_mean='mean', response_time_median='median').reset_index().set_index("item_id")
        self.stu_answer_table = train_set.pivot_table(index='student_id', columns='item_id', values='response')
        self.stu_ability_table = train_set[["student_id", "ability_class"]].drop_duplicates().reset_index(drop=True).set_index("student_id")

        assert model in ["c45", "id3", "cart"], "Only c45, id3, cart are available"
        self.model = model
        self.time_aware = time_aware
        self.mean_or_median = mean_or_median

        self.root_q = None


    def __entropy(self, current_set):
        '''
        Compute the entropy of the ability class given the answered questions, where
            - p_i = |S_i| / |S|
            - entropy = \sum_i - p_i * log_2(p_i)

        Parameter:
            - current_set (pd df)
        
        Return:
            - entropy (float)
        '''

        current_ability_set = self.stu_ability_table.loc[current_set.index, "ability_class"].values
        prob = np.bincount(current_ability_set) / len(current_ability_set)
        entropy = -np.sum(prob * np.log2(np.where(prob > 0, prob, 1)))
        
        return entropy
    

    def __gini(self, current_response_table):
        
        current_ability_set = self.stu_ability_table.loc[current_response_table.index]["ability_class"].values
        prob = np.bincount(current_ability_set) / len(current_ability_set)
        gini = 1 - np.sum(prob ** 2)

        return gini


    def __split_info(self, left_prob, right_prob):
        '''
        Compute the split info:

        Parameter:
            - left_prob (float)
            - right_prob (float)
        
        Return:
            - SI (float)
        '''

        return - left_prob * np.log2(np.where(left_prob > 0, left_prob, 1)) - right_prob * np.log2(np.where(right_prob > 0, right_prob, 1))
    

    def __compute_IG(self, item_id, current_set):
        '''
        Compute IG for each question, given current_set
        
        Parameters:
            - item_id (str): the question to be evaluated
            - current_set (pd df)

        Returns:
            - IG (float)
        '''

        if len(current_set) == 0:
            return -np.inf

        current_set_left = current_set[current_set[item_id] == 0]
        current_set_right = current_set[current_set[item_id] == 1]

        if len(current_set_left) == 0 or len(current_set_right) == 0:
            return -np.inf

        left_prob = len(current_set_left) / len(current_set)
        right_prob = len(current_set_right) / len(current_set)

        IG = self.__entropy(current_set) - left_prob * self.__entropy(current_set_left) - right_prob * self.__entropy(current_set_right)

        return IG


    def __compute_gain_ratio(self, item_id, current_set):
        '''
        Compute IG and SI for each question, given current_set
        
        Parameters:
            - item_id (str): the question to be evaluated
            - current_set (pd df)

        Returns:
            - IG/SI (float)
        '''

        if len(current_set) == 0:
            return -np.inf

        current_set_left = current_set[current_set[item_id] == 0]
        current_set_right = current_set[current_set[item_id] == 1]

        if len(current_set_left) == 0 or len(current_set_right) == 0:
            return -np.inf

        left_prob = len(current_set_left) / len(current_set)
        right_prob = len(current_set_right) / len(current_set)

        IG = self.__entropy(current_set) - left_prob * self.__entropy(current_set_left) - right_prob * self.__entropy(current_set_right)
        SI = self.__split_info(left_prob, right_prob)

        return IG/SI
    

    def __compute_gini(self, item_id, current_response_table):

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
    

    def __compute_time(self, item_id):
        return self.q_bank.loc[item_id]["response_time_mean"] if self.mean_or_median == "mean" else self.q_bank.loc[item_id]["response_time_median"]


    def __select_best_q(self, qs_available, current_set):
        '''
        Selects the best question (based on the questions answered so far)

        Parameters:
            - qs_available (pd df): with column "item_id"
            - current_set (pd df): with index "student_id", column each question item_id
        
        Returns
            - best_q["item_id"] (str)
            - best_q["response_time"] (float): avg response time (can be changed into avg over the students that satisfy current_set)
        '''

        if current_set.empty:
            return None, None

        best_idx = None
        if self.model == "c45":
            if not self.time_aware :
                qs_IG = np.array([self.__compute_gain_ratio(item_id, current_set) for item_id in qs_available.index])
                best_idx = np.argmax(qs_IG)
            else:
                qs_IG_T = np.array([self.__compute_gain_ratio(item_id, current_set) / self.__compute_time(item_id) for item_id in qs_available.index])
                best_idx = np.argmax(qs_IG_T)
        
        elif self.model == "id3":
            if not self.time_aware:
                IG = np.array([self.__compute_IG(item_id, current_set) for item_id in qs_available.index])
                best_idx = np.argmax(IG)
            else:
                IG_T = np.array([self.__compute_IG(item_id, current_set) / self.__compute_time(item_id) for item_id in qs_available.index])
                best_idx = np.argmax(IG_T)

        elif self.model == "cart":
            if not self.time_aware:
                qs_gini = np.array([self.__compute_gini(item_id, current_set) for item_id in qs_available.index])
                best_idx = np.argmin(qs_gini)
            else:
                qs_gini = np.array([self.__compute_gini(item_id, current_set) / self.__compute_time(item_id) for item_id in qs_available.index])
                best_idx = np.argmin(qs_gini)       # or maximize gini gain

        best_q_id = qs_available.index[best_idx]
        best_q_response_time = qs_available.iloc[best_idx]["response_time_mean"] if self.mean_or_median == "mean" else qs_available.iloc[best_idx]["response_time_median"]
        
        return best_q_id, best_q_response_time



    def __build_dt(self, root_q):
        '''
        Build a (sub-)tree rooted at root_q

        Parameter:
            - root_q (QuestionNode)
        '''

        if root_q is None:
            return
        if root_q.num_qs_so_far >= 15 or root_q.time_remain <= 0:
            return
        if root_q.qs_available.empty:
            return

        current_set_left = root_q.current_set[root_q.current_set[root_q.item_id] == 0]
        current_set_right = root_q.current_set[root_q.current_set[root_q.item_id] == 1]

        start = time.time()
        left_item_id, left_time_taken = self.__select_best_q(qs_available=root_q.qs_available.copy(),
                                                             current_set=current_set_left)
        right_item_id, right_time_taken = self.__select_best_q(qs_available=root_q.qs_available.copy(),
                                                               current_set=current_set_right)
        end = time.time()
        # print(f"{end-start:.6f}")
        
        left_q, right_q = None, None
        if left_item_id is not None:
            time_remain_left = root_q.time_remain-left_time_taken

            # remove the selected question
            qs_available_left = root_q.qs_available.drop(index=left_item_id)

            # remove the questions that take longer than the time remaining
            time_choice = "response_time_mean" if self.mean_or_median else "response_time_median"
            qs_available_left = qs_available_left[qs_available_left[time_choice] < time_remain_left]

            left_q = QuestionNode(item_id=left_item_id,
                                   time_taken=left_time_taken,
                                   time_remain=time_remain_left,
                                   num_qs_so_far=root_q.num_qs_so_far+1,
                                   current_set=current_set_left,
                                   qs_available=qs_available_left,
                                   left=None,
                                   right=None)
            
        if right_item_id is not None:
            time_remain_right = root_q.time_remain-right_time_taken

            # remove the selected question
            qs_available_right = root_q.qs_available.drop(index=right_item_id)

            # remove the questions that take longer than the time remaining
            time_choice = "response_time_mean" if self.mean_or_median else "response_time_median"
            qs_available_right = qs_available_right[qs_available_right[time_choice] < time_remain_right]

            right_q = QuestionNode(item_id=right_item_id,
                                   time_taken=right_time_taken,
                                   time_remain=time_remain_right,
                                   num_qs_so_far=root_q.num_qs_so_far+1,
                                   current_set=current_set_right,
                                   qs_available=qs_available_right,
                                   left=None,
                                   right=None)

        root_q.left = left_q
        root_q.right = right_q

        self.__build_dt(root_q.left)
        self.__build_dt(root_q.right)


    def train(self):
        '''
        Train (Generate) a decision tree based on train_set
        '''

        # choose the first question
        item_id, time_taken = self.__select_best_q(current_set=self.stu_answer_table,
                                                   qs_available=self.q_bank.copy())

        time_remain_root = MAX_TIME-time_taken

        # remove the selected question
        qs_available = self.q_bank.drop(index=item_id)

        # remove the questions that take longer than the time remaining
        time_choice = "response_time_mean" if self.mean_or_median else "response_time_median"
        qs_available = qs_available[qs_available[time_choice] < time_remain_root]

        self.root_q = QuestionNode(item_id=item_id,
                                   time_taken=time_taken,
                                    time_remain=time_remain_root,
                                    num_qs_so_far=1,
                                    current_set=self.stu_answer_table,
                                    qs_available=qs_available,
                                    left=None,
                                    right=None)
        self.__build_dt(self.root_q)
        self.__shrink_tree(self.root_q)
        self.q_bank = None
        self.stu_answer_table = None
        self.stu_ability_table = None


    def __shrink_tree(self, current_node):
        if current_node is not None:
            current_node.current_set = None
            current_node.qs_available = None

            self.__shrink_tree(current_node.left)
            self.__shrink_tree(current_node.right)



if __name__ == "__main__":
    pass
