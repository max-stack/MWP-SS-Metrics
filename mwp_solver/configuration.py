# @Author: Max Wilson-Hebben

import sys
import os
from logging import getLogger
from enum import Enum
import json
from pathlib import Path

import torch

from constants import LOG_PATH, CHECKPOINT_PATH, TRAINED_MODEL_PATH, DATASET_PATH
from utils.utils import read_json_data, get_model

# generates a config object that contains parameters used in training and testig the model
class Config:
    def __init__(self, model_name=None, config_dict={}):
        self.model_name = model_name
        self.orig_config = {}
        self.model_config = {}
        self.dataset_config = {}
        self.final_config = config_dict

        self.get_orig_config()
        self.get_model_config()
        self.create_dataset_config()
        self.create_final_config()
        self.set_task_type()
        self.set_paths()
        self.set_model()
        self.init_device()
    
    def get_orig_config(self):
        config_file = open(Path(__file__).parent / "configs/config.json")
        self.orig_config = json.load(config_file)

    def get_model_config(self):
        config_file = open(Path(__file__).parent / "configs/{}.json".format(self.model_name))
        self.model_config = json.load(config_file)
    
    def create_dataset_config(self):
        config_file = open(Path(__file__).parent / "configs/datset.json")
        self.dataset_config = json.load(config_file)
    
    def create_final_config(self):
        self.final_config.update(self.orig_config)
        self.final_config.update(self.model_config)
        self.final_config.update(self.dataset_config)
    
    def set_task_type(self):
        if self.final_config["task_type"] == "single_equation":
            self.final_config["single"] = True
        else:
            self.final_config["single"] = False
    
    def set_model(self):
        self.final_config["model"] = self.model_name
    
    def set_paths(self): 
        self.final_config["root"] = str(Path(__file__).parent)
        self.final_config["log_path"] = LOG_PATH + "/{}_logs.log".format(self.model_name)
        self.final_config["checkpoint_path"] = CHECKPOINT_PATH + "/{}_checkpoint.pth".format(self.model_name)
        self.final_config["trained_model_path"] = TRAINED_MODEL_PATH + "/{}_trained_model.pth".format(self.model_name)
        self.final_config["dataset_path"] = DATASET_PATH
    
    def init_device(self):
        if self.final_config["gpu_id"] == None:
            if torch.cuda.is_available() and self.final_config["use_gpu"]:
                self.final_config["gpu_id"] = "0"
            else:
                self.final_config["gpu_id"] = ""
        else:
            if self.final_config["use_gpu"] != True:
                self.final_config["gpu_id"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config["gpu_id"])
        self.final_config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.final_config["map_location"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.final_config['gpu_nums'] = torch.cuda.device_count()
    
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Index must be a string")
        self.final_config[key] = value

    def __getitem__(self, item):
        if item in self.final_config:
            return self.final_config[item]
        else:
            return None

    def __str__(self):
        args_info = ''
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config.items()])
        args_info += '\n\n'
        return args_info
