# MWP-SS-Metrics

## Introduction to the Package

The MWP-SS-Metrics is a metrics package used to evaluate the performance of math word problem generation models. The model generates 10 high quality metrics of evaluation. It has two main tests, one for testing the similarity of generated problems to the training dataset and one for looking at the solvability of generated problems. A more in-depth discussion on the package is presented in the report "Evaluating Math Word Problem Generation Techniques" 

## How to Run

To run the package, clone the repository and find the "run_metrics.py" file in the "metrics" folder. This is the only class that should be used when running the package. There are two functions associated with the "Run_Metrics" class. "init_model" and "get_metrics". The "init_model" method should be used to train the MWP solver models, given the train, test and validation dataset split that was used to generate the math word problems. This will likely take quite a large amount of time since two MWP solver models must be trained on the data. The "get_metrics" method can be used following the use of the "init_model" method. This method will calculate the similarity of problems to the trainset, re-run the solver models excluding highly similar problems and output the final metric results.

## Parameters

## Interpretation of Metrics

## Changing More Advanced Features

## Requirements

