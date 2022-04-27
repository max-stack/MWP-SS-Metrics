# MWP-SS-Metrics

## Introduction to the Package

The MWP-SS-Metrics is a metrics package used to evaluate the performance of math word problem generation models. The model generates 10 high quality metrics of evaluation. It has two main tests, one for testing the similarity of generated problems to the training dataset and one for looking at the solvability of generated problems. A more in-depth discussion on the package is presented in the report "Evaluating Math Word Problem Generation Techniques". The package was written using Python 3 and has been run successfully on Python 3.8.

## How to Run

To run the package, clone the repository and find the "run_metrics.py" file in the "metrics" folder. This is the only class that should be used when running the package. There are two functions associated with the "Run_Metrics" class. "init_model" and "get_metrics". The "init_model" method should be used to train the MWP solver models, given the train, test and validation dataset split that was used to generate the math word problems. This will likely take quite a large amount of time since two MWP solver models must be trained on the data. The "get_metrics" method can be used following the use of the "init_model" method. This method will calculate the similarity of problems to the trainset, re-run the solver models excluding highly similar problems and output the final metric results.

An more thorough look at parameters is present in the next section.

## Parameters

**Run_Metrics Class** - To create an instance of the Run_Metrics class, there is one forced parameter "language" and six optional parameters. Each is detailed below. The first four are split into Graph2Tree solvability and SAUSolver solvability, using both models separately to make the evaluation. More detail on the metrics can be found in the report "Evaluating Math Word Problem Generation Techniques".

- **language** - specifies the language to use. Two letters should be used such as "en" or "zh", as specified by NLTK documentation.
- **task_type** - used to specify the type of equations, either "single_equation" or "multi_equation". Default value is "single_equation".
- **epoch_nums** - the number of epochs that the MWP solver models will run for. Default value is 15.
- **linear** - if non-linear equations feature in the training set, this should be set to False.
- **use_gpu** - if this is set to false, only the CPU will be used in training the MWP solver models.
- **source_equation_fix** - set to "infix" by default. "postfix" and "prefix" can also be specified if necessary.
- **train_batch_size** - specifies the batch size for training. By default it is set to 32.

**init_model() Method** - There are three forced parameters associated with this model, each should be a list with 3 inner lists, containing the problem texts, equations and answers of each problem in the dataset respectively. These should be the train, test and valid set used to train the MWP generation model.

**get_metrics() Method** - This takes two parameters. "test_data" should be a list of 3 inner lists but containing the generated MWPs dataset. "similar_ratio" is as the threshold below which math word problems are exlcuding from the "Exclusion" tests. This is set to 0.9 by default.

## Interpretation of Metrics
There are 10 associated metrics output from the package. They are detailed below.

- **Solvability of Problems in Original Dataset** - The accuracy of the trained MWP solver models on the test set associated with the original dataset.
  - Graph2Tree
  - SAUSolver

- **Solvability of Problems in Original Dataset Excluding High Similarity Problems** - Same as above but after excluding problems with SImilarity Ration > 0.9 from the dataset.
  - Graph2Tree
  - SAUSolver

- **Solvability of Generated Problems** - The accuracy of the trained MWP solver models on the model generated dataset. 
  - Graph2Tree
  - SAUSolver

- **Solvability of Generated Problems Excluding High Similarity Problems** Same as above but after excluding problems with SImilarity Ration > 0.9 from the dataset.
  - Graph2Tree
  - SAUSolver

- **Average Similarity Score of Problems in Original Dataset** - The similarity score is calculated for each problem in the test set associated with the original dataset and the average over the dataset is calculated.

- **Average Similarity Score of Generated Problems** - The similarity score is calculated for each problem in the model generated dataset and the average over the dataset is calculated.


## Changing More Advanced Features

To change more advanced features such as the number of layers in the MWP solver neural networks or the number of nodes in each layer, these have to be done in the code. Most of the configuration takes place under the "configs" subfolder in the "mwp_solver" folder.

