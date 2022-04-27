# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:31:08
# @File: quick_start.py
# Modified by Max Wilson-Hebben

import os
import sys
from logging import getLogger

import torch

from configuration import Config
from constants import MODELS
from evaluate.evaluator import AbstractEvaluator, InfixEvaluator, PostfixEvaluator, PrefixEvaluator, MultiWayTreeEvaluator
from evaluate.evaluator import MultiEncDecEvaluator
from utils.data_utils import create_dataset, create_dataloader
from utils.utils import get_model, init_seed, get_trainer
from utils.enum_type import SpecialTokens, FixType
from utils.logger import init_logger

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

#Function updated by Max Wilson-Hebben
def run_toolkit(model_name, config_dict={}, test_only=False):
    if(model_name not in MODELS):
        raise Exception("Model {} not available in MODELS".format(model_name))
        
    config = Config(model_name, config_dict)

    init_seed(config['random_seed'], True)

    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config, test_only)

    dataset.dataset_load()
    
    dataloader = create_dataloader(config)(config, dataset)

    model = get_model(config["model"])(config,dataset).to(config["device"])
    
    if config["equation_fix"] == FixType.Prefix:
        evaluator = PrefixEvaluator(config)
    elif config["equation_fix"] == FixType.Nonfix or config["equation_fix"] == FixType.Infix:
        evaluator = InfixEvaluator(config)
    elif config["equation_fix"] == FixType.Postfix:
        evaluator = PostfixEvaluator(config)
    elif config["equation_fix"] == FixType.MultiWayTree:
        evaluator = MultiWayTreeEvaluator(config)
    else:
        raise NotImplementedError
    
    if config['model'].lower() in ['multiencdec']:
        evaluator = MultiEncDecEvaluator(config)

    trainer = get_trainer(config)(config, model, dataloader, evaluator)
    logger.info(model)
    if test_only:
        return trainer.test()
    else:
        return trainer.fit()