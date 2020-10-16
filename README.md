## Installation and setup
In order to run things as intended, some setup is required. The setup is automated, but requires make and conda. Once installed, you can just run ``$ make`` to install everything else. This will create some githooks, some empty folders, and new conda environments. 

There are two conda environments: `mim` and `mim-gpu`. The only difference between them is that one installs tensorflow without GPU support, and the other installs with GPU support. The script will prompt you to install one or both of them. If you later need to update the environments, you can run that part of the make script again with `$ make conda`

## Githooks
There are two git-hooks currently in place (see `.hooks/`), one for pre-commit and one for pre-push. The pre-commit hook will run flake8 for all staged changes and reject the commit if the pep8 standard is not fulfilled. The hook will ignore unstaged changes or files that haven't changed.

The pre-push hook will run pytest to verify that all the tests pass before any code can be pushed. If any of the tests fail, the push will be denied. Pushing is also denied, without running the tests, if there are uncommitted changes in staged files. This is to make sure that the tests test the code that would be pushed. If you want to keep uncommitted changes and still push, stash the changes first with `git stash`.

## Experiments
### Definition
In the context of this repository, an experiment is a data structure that specifies all the details needed to select and prepare data and use it to run some classification or regression task. It is a way of specifying exactly which data is used on which algorithm in a readable and reproducible way. Multiple experiments are grouped together in enums and can be run individually or all at once with a single line of code. The results of an experiment is saved to file and can be investigated with the help of a presenter module, which aids in comparing multiple similar experiments. 

### Motivation
The proposed structure is intended to solve three problems:

- Reproducibility
- Documentation
- Exploration

Often it is easy to forget, later on, exactly what has been tested and what the results were. If you find a bug somewhere, you want to be able to check whether the bug affected any past experiments, and if so, re-run those experiments. And it should be easy to test and compare the results of multiple different ideas. 

### Examples
Experiments are defined in the `mim.experiments` module, where they are grouped into different files and enums depending on their purpose. 

To run an experiment, use `mim.experiments.run`, like this:
```bash
python -m mim.experiments.run MyocardialInfarction
```

This will run all the experiments in the MyocardialInfarction enum. You can also specify individual experiments as well as rerun experiments that already have been run before. For a complete list of arguments and what they do, try
```bash
python -m mim.experiments.run --help
```

Once an experiment has finished running, the results can be examined and compared with those of other experiments. This is done in the Presenter class in `mim.presenter`. Here's a small example where all the test results from MyocardialInfarction will be printed out in a nice table:

```
> from mim.presenter import Presenter
> p = Presenter('MyocardialInfarction')
> p.describe()
                 test_score  test_score_std  train_score    commit  changed                   timestamp
THAN_EXPECT_GB     0.952053        0.004228     0.960708  b23c7c8e    False  2020-10-16 13:54:04.269414
THAN_EXPECT_RF     0.944819        0.005433     1.000000  b23c7c8e    False  2020-10-16 13:54:19.054416
THAN_EXPECT_RF2    0.951577        0.005644     1.000000  b23c7c8e    False  2020-10-16 13:54:32.071905
PTBXL_SMALL        0.490661        0.046217     0.523585  b23c7c8e    False  2020-10-16 13:54:34.728484
```