# @Author: Max Wilson-Hebben

import sys
sys.path.insert(0, '../mwp_solver')
from run_solver import MWP_Solver
sys.path.insert(0, '../similarity_test')
from similarity_metric import Similarity_Metric
from test_unsimilar import Test_Unsimilar

class Run_Metrics:
    def __init__(self, language, task_type="single_equation", epoch_nums=15, linear=True, use_gpu=True, source_equation_fix="infix", train_batch_size=32):
        self.language = language
        self.task_type = task_type
        self.epoch_nums = epoch_nums
        self.linear = linear
        self.task_type = self.task_type
        self.use_gpu = use_gpu
        self.source_equation_fix = source_equation_fix
        self.train_batch_size = train_batch_size
        self.config_dict = self.create_config_dict()

    def init_model(self, train_data, test_data, valid_data):
        solver = MWP_Solver(self.config_dict)
        self.train_metric = solver.train_solver(train_data, test_data, valid_data)
    
    def get_metrics(self, test_data, similar_ratio=0.9):
        test_unsimilar_orig = Test_Unsimilar(config_dict=self.config_dict, similar_ratio=similar_ratio)
        test_unsimilar_orig_metric = test_unsimilar_orig.test_unsimilar()

        similarity_orig = Similarity_Metric()
        similarity_orig_metric = similarity_orig.average_similarity()

        test_solver = MWP_Solver(self.config_dict)
        test_metric = test_solver.test_solver(test_data)

        similarity_gen = Similarity_Metric()
        similarity_gen_metric = similarity_gen.average_similarity()
        
        test_unsimilar_gen = Test_Unsimilar(config_dict=self.config_dict, similar_ratio=similar_ratio)
        test_unsimilar_gen_metric = test_unsimilar_gen.test_unsimilar()
        
        final_metric = {}
        final_metric["Solvability of Problems in Original Dataset"] = self.train_metric
        final_metric["Solvability of Problems in Original Dataset"]["Excluding High Similarity Problems"] = test_unsimilar_orig_metric
        final_metric["Solvability of Generated Problems"] = test_metric
        final_metric["Solvability of Generated Problems"]["Excluding High Similarity Problems"] = test_unsimilar_gen_metric
        final_metric["Average Similarity Score of Problems in Original Dataset"] = similarity_orig_metric["similarity_score"]
        final_metric["Average Similarity Score of Generated Problems"] = similarity_gen_metric["similarity_score"]
        
        return final_metric
    
    def create_config_dict(self):
        config_dict = {}
        config_dict["language"] = self.language
        config_dict["task_type"] = self.task_type
        config_dict["epoch_nums"] = self.epoch_nums
        config_dict["linear"] = self.linear
        config_dict["use_gpu"] = self.use_gpu
        config_dict["source_equation_fix"] = self.source_equation_fix
        config_dict["train_batch_size"] = self.train_batch_size

        return config_dict
