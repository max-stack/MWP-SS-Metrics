# @Author: Max Wilson-Hebben

from difflib import SequenceMatcher
import json
import sys
from similarity_ratio import Similarity_Ratio
sys.path.insert(0, '../mwp_solver')
from run_solver import MWP_Solver
from constants import TRAINSET_PATH, TESTSET_PATH, VALIDSET_PATH


# used to find the solvability of a dataset, excluding problems with high similarity score.
# similarity_ratio is used to determine the limit below which problems are excluded.
class Test_Unsimilar:
    def __init__(self, config_dict, similar_ratio=0.9):
        self.similar_ratio = similar_ratio
        self.config_dict = config_dict
    
    def test_unsimilar(self):
        self.remove_similar()
        solver = MWP_Solver(self.config_dict)
        test_results = solver.test_solver(set_created=True)

        results = {}
        
        results["Graph2Tree"] = test_results["Graph2Tree"]
        results["SAUSolver"] = test_results["SAUSolver"]

        return results

    def remove_similar(self):
        new_testset = []

        with open(TESTSET_PATH, encoding="utf-8") as test_file:
            test_data = json.load(test_file)
        
        with open(TRAINSET_PATH, encoding="utf-8") as train_file:
            train_data = json.load(train_file)
        
        train_questions = [x["segmented_text"] for x in train_data]

        for test_question in test_data:
            test_nl = test_question["segmented_text"]
            max_ratio = self.similarity_score(test_nl, train_questions)
            if max_ratio < self.similar_ratio:
                new_testset.append(test_question)

        with open(TESTSET_PATH, "w", encoding="utf-8") as testset_file:
            json.dump(new_testset, testset_file, ensure_ascii=False, indent=4)

    def similarity_score(self, problem1, train_data):
        highest_similarity = 0
        for j in range(len(train_data)):
            similarity_ratio = Similarity_Ratio(problem1, train_data[j])
            similarity_value = similarity_ratio.cosine_similarity()
            if similarity_value > highest_similarity:
                highest_similarity = similarity_value

        return highest_similarity