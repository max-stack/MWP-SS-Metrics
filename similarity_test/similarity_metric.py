# @Author: Max Wilson-Hebben

from difflib import SequenceMatcher
import sys
import json
from similarity_ratio import Similarity_Ratio
sys.path.insert(0, '../mwp_solver')
from constants import TRAINSET_PATH, TESTSET_PATH, VALIDSET_PATH

# used to generate the average similarity score over problems in a dataset
class Similarity_Metric:
    def __init__(self):
        pass

    def get_nl(self, file):
        question_list = []
        with open(file, encoding="utf-8") as json_file:
            data = json.load(json_file)

        for question in data:
            question_list.append(question["segmented_text"])
        
        return question_list

    def similarity_score(self, problem1, train_data):
        highest_similarity = 0
        for j in range(len(train_data)):
            similarity_ratio = Similarity_Ratio(problem1, train_data[j])
            similarity_value = similarity_ratio.cosine_similarity()
            if similarity_value > highest_similarity:
                highest_similarity = similarity_value

        return highest_similarity

    def average_similarity(self):
        test_nl = self.get_nl(TESTSET_PATH)
        train_nl = self.get_nl(TRAINSET_PATH)

        added_similarities = 0
        for test_question in test_nl:
            max_ratio = self.similarity_score(test_question, train_nl)
            added_similarities += max_ratio
        
        result = {}

        result["similarity_score"] = added_similarities / len(test_nl)

        return result
