# @Author: Max Wilson-Hebben

from quick_start import run_toolkit
from create_datasets import Create_Datasets
import os
from os.path import exists
from constants import DEPREL_TREE_PATH

# used to run the training or testing model separately
class MWP_Solver:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def train_solver(self, train_data, test_data, valid_data):
        dataset_creator = Create_Datasets(test_data, valid_data, train_data)
        dataset_creator.create_train_data()
        dataset_creator.create_test_data()
        dataset_creator.create_valid_data()

        results = {}
        
        self.remove_deprel_tree()
        results["Graph2Tree"] = run_toolkit("Graph2Tree", self.config_dict)
        results["SAUSolver"] = run_toolkit("SAUSolver", self.config_dict)

        return results

    def test_solver(self, test_data=None, set_created=False):
        if(not set_created):
            dataset_creator = Create_Datasets(test_data)
            dataset_creator.create_test_data()

        results = {}
        
        self.remove_deprel_tree()
        results["Graph2Tree"] = run_toolkit("Graph2Tree", self.config_dict, test_only=True)
        results["SAUSolver"] = run_toolkit("SAUSolver", self.config_dict, test_only=True)

        return results
    
    def remove_deprel_tree(self):
        if exists(DEPREL_TREE_PATH):
            os.remove(DEPREL_TREE_PATH)
