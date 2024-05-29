# Mim
Welcome to the mim repository. For specific articles, please refer to the articles section below.

## Purpose and motivation
When exploring and developing machine learning models it can sometimes be difficult to remember what was tested and what the results were. If a mistake is found somewhere in the code, it can be hard to figure out what, if any, previous experiments were affected, and re-running them can be a challenge. 

This purpose of this repository is to help with some of these issues. In particular, the project is intended to help with exploration, documentation and reproducibility. 

A typical machine learning project starts with a dataset, which gets processed into a set of features and labels (usually denoted x and y), which are then plugged into a model that is fitted or trained to predict the labels based on the features. 

## Core concepts
In the context of this repository, an Experiment is a data structure that specifies all the details needed to select and prepare data and use it to run some classification or regression task. It is a way of specifying exactly which data is used on which algorithm in a readable and reproducible way. Multiple experiments are grouped together in enums and can be run individually or all at once with a single line of code. The results of an experiment is saved to disk and can be investigated with the help of a presenter module, which aids in comparing multiple similar experiments.

The Extractor is a data structure that helps to load and prepare all the data needed to run an experiment. Feature engineering can be parameterised to make it easier to test different variations of experiments, as needed.

The repository is divided into different modules, as follows:
* The mim module contains generic building blocks that can be useful for multiple projects. It defines the Experiment, Extractor and Model, classes, among others, and it also contains the main function from which all experiments are run. 
* The massage module contains code for cleaning and preparing data from specific data sets. Since one dataset can be used for several projects, separating the code for preparing data from individual projects is useful for code reuse. 
* The projects module contains a sub-module for each *project*. A project is a collection of experiments intended to answer some research question. A project will typically define its own Extractor and Experiment classes. The Extractor helps to parameterise the feature engineering, making use of code from the massage module, while the Experiment class helps to enumerate and specify the exact models, parameters and input data to test.

## Installation and setup
In order to run things as intended, some setup is required. The setup is automated, but requires make and conda. Once installed, you can just run ``$ make`` to install everything else. This will create some githooks, some empty folders, and new conda environments. 

The pre-commit hooks require a programmed called pre-commit, which is included in the conda environment. Thus, you need to first make the conda environment, then activate it (conda activate mim), and then you can make the hooks with ``$ make hooks``.

There are two conda environments: `mim-cpu` and `mim-gpu`. The only difference between them is that one installs tensorflow without GPU support, and the other installs with GPU support. The script will prompt you to install one or both of them. If you later need to update the environments, you can run that part of the make script again with `$ make conda`

## Githooks
There are two git-hooks currently in place (see `.hooks/`), one for pre-commit and one for pre-push. The pre-commit hook will run flake8 for all staged changes and reject the commit if the pep8 standard is not fulfilled. The hook will ignore unstaged changes or files that haven't changed.

The pre-push hook will run pytest to verify that all the tests pass before any code can be pushed. If any of the tests fail, the push will be denied. Pushing is also denied, without running the tests, if there are uncommitted changes in staged files. This is to make sure that the tests test the code that would be pushed. If you want to keep uncommitted changes and still push, stash the changes first with `git stash`.

# Articles

## Prior ECGs not useful for machine learning predictions of MACE in ED chest pain patients
Link to the article: https://doi.org/10.1016/j.jelectrocard.2023.11.002

All the code as well as further instructions can be found in the projects/serial_ecgs folder.

## Transfer Learning for Predicting Acute Myocardial Infarction Using Electrocardiograms
Paper under review. The code is found in the projects/transfer folder.
