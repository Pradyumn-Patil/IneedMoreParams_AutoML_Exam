# AutoML Exam - SS25 (Text Data)

This repository contains a comprehensive AutoML system for text classification developed for the SS25 AutoML course
at the University of Freiburg. The system automatically selects and optimizes machine learning pipelines 
for text classification tasks.

## Features

- **Multiple Approaches**: Logistic Regression, Feed-Forward Neural Networks, LSTM, CNN-LSTM Hybrid, Transformer models
- **Advanced Text Processing**: Enhanced TF-IDF with trigrams and character n-grams, advanced preprocessing
- **Hyperparameter Optimization**: Multi-fidelity HPO with Optuna (TPE, Random, CMA-ES samplers)
- **Neural Architecture Search**: Automated architecture selection for neural models
- **Text Augmentation**: Synonym replacement, random insertion/deletion/swap
- **Meta-Learning**: Dataset similarity analysis for configuration warm-starting
- **Ensemble Methods**: Voting, stacking, and weighted averaging
- **One-Click Solution**: Fully automated pipeline with intelligent resource allocation

## Installation

To install the repository, first create an environment of your choice and activate it. 

For example, using `venv`:

You can change the python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-text-env
source automl-text-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-text-env python=3.10
conda activate automl-text-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

*NOTE*: this is an editable install which allows you to edit the package code without requiring re-installations.

You can test that the installation was successful by running the following command:

```bash
python -c "import automl; print(automl.__file__)"
# this should print the full path to your cloned install of this repo
```

We make no restrictions on the python library or version you use, but we recommend using python 3.10 or higher.

## Code

The system provides multiple entry points for different use cases:

### Main Scripts

* `run_automl_complete.py`: **One-click complete AutoML solution** that automatically selects the best approach, allocates budget, and runs the full pipeline
* `run.py`: Basic training script with manual configuration options
* `run_with_config.py`: Run experiments using YAML configuration files
* `run_all_experiments.py`: Batch runner for all datasets with optimized settings

### Core Package Structure

* `automl/core.py`: Main TextAutoML class orchestrating the entire pipeline
* `automl/models.py`: Neural network architectures (FFNN, LSTM, CNN-LSTM, NAS-searchable models)
* `automl/datasets.py`: Dataset loaders for all supported text classification datasets
* `automl/preprocessing.py`: Advanced text preprocessing with dataset-specific strategies
* `automl/augmentation.py`: Text augmentation techniques for improving model robustness
* `automl/ensemble.py`: Ensemble methods for combining multiple models
* `automl/meta_learning.py`: Meta-learning for dataset similarity and warm-starting

### Quick Start Examples

```bash
# One-click solution with 24-hour budget
python run_automl_complete.py --dataset amazon --data-path ./data --budget 24

# Quick test with 1-hour budget
python run_automl_complete.py --dataset amazon --data-path ./data --quick-test

# Basic training with specific approach
python run.py --data-path ./data --dataset amazon --approach logistic --epochs 5

# With hyperparameter optimization
python run.py --data-path ./data --dataset amazon --use-hpo --hpo-trials 50
```


## Data

We selected 4 different text-classification datasets which you can use to develop your AutoML system and we will provide you with 
a test dataset to evaluate your system at a later point in time. 

The dataset can be automatically or programatically downloaded and extracted from: [https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase1.zip](https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase1.zip)

The downloaded datasets should have the following structure:
```bash
<target-folder>
├── ag_news
│   ├── train.csv
│   ├── test.csv
├── amazon
│   ├── train.csv
│   ├── test.csv
├── imdb
│   ├── train.csv
│   ├── test.csv
├── dbpedia
│   ├── train.csv
│   ├── test.csv
```

### Meta-data for datasets:

The following table will provide you an overview of their characteristics and also a reference value for the test accuracy.
*NOTE*: These scores were obtained through a rather simple HPO on a crudely constructed search space, for an undisclosed HPO budget and compute resources.

| Dataset Name | Labels | Rows | Seq. Length: `min` | Seq. Length: `max` | Seq. Length: `mean` | Seq. Length: `median` | Reference Accuracy |
| --- | --- |  --- |  --- |  --- | --- | --- | --- |
| amazon | 3 | 24985 | 4 | 15521 | 512 | 230 | 81.799% |
| imdb | 2 | 25000 | 52 | 13584 | 1300 | 962 | 86.993% |
| ag_news | 4 | 120000 | 99 | 1012 | 235 | 231 | 90.265% |
| dbpedia | 14 | 560000 | 11 | 13573 | 300 | 301 | 97.882% |
| *final\_exam\_dataset* | TBA | TBA | TBA | TBA | TBA | TBA | TBA |

*NOTE*: sequence length calculated at the raw character level

## Performance Results

Our AutoML system achieves the following test accuracies:

| Dataset | Baseline | Our Result | Improvement |
|---------|----------|------------|-------------|
| Amazon  | 81.799%  | 83.58%     | +1.78%      |
| IMDB    | 86.993%  | 84.89%     | -2.10%      |
| AG News | 90.265%  | 87.38%     | -2.89%      |
| DBpedia | 97.882%  | 95.50%     | -2.38%      |

*Note: Results obtained with limited computational resources. Full 24-hour budget expected to improve performance.*

We will add the test dataset later in the final Github Classroom template code that will be released.
 <!-- by pushing its class definition to the `datasets.py` file.  -->
The test dataset will be in the same format as the training datasets, but `test.csv` will only contain `nan`'s for labels.


## Running an initial test

After having downloaded and extracted the data at a suitable location, this is the parent data directory. \\
To run a quick test:

```bash
python run.py \
  --data-path <path-to-data-parent> \
  --dataset amazon \
  --epochs 1 \
  --data-fraction 0.2
```
*TIP*: play with the batch size and different approaches for an epoch (or few mini-batches) to estimate compute requirements given your hardware availability.

You are free to modify these files and command line arguments as you see fit.

<!-- ## Final submission

The final test predictions should be uploaded in a file `final_test_preds.npy`, with each line containing the predictions for the input in the exact order of `X_test` given.

Upload your poster as a PDF file named as `final_poster_text_<team-name>.pdf`, following the template given [here](https://docs.google.com/presentation/d/1T55GFGsoon9a4T_oUm4WXOhW8wMEQL3M/edit?usp=sharing&ouid=118357408080604124767&rtpof=true&sd=true). -->

## Tips

* If you need to add dependencies that you and your teammates are all on the same page, you can modify the
  `pyproject.toml` file and add the dependencies there. This will ensure that everyone has the same dependencies

* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
  predictions, etc. Also, be friendly teammate and ignore your virtual environment and any additional folders/files
  created by your IDE.
