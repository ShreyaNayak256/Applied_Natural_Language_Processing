# NLP Model Execution Guide for CoNLL 2003 Task

This guide provides instructions on how to execute the scripts `SN_NLP_HW4_task1.py` and `SN_NLP_HW4_task2.py`. 
Both scripts are written for Python 3.11.

## Prerequisites

Before running the scripts, please ensure the following files are present in the same directory as the scripts:

- `conlleval.py` - Evaluation script provided for assessing the model's performance.
- `model_hw4_task1.pth` - Saved PyTorch model for Task 1.
- `model_hw4_task2.pth` - Saved PyTorch model for Task 2 (required only for Task 2).
- `glove.6B.100d.txt` - GloVe embeddings file (required only for Task 2).
- `model_hw4_task3.pth` - Saved PyTorch model for Task 3 (required only for Task 3).

The Python library `datasets` should also be pre-installed in the environment.
if not, please install using the following command: 
pip install datasets

## Running the Scripts:

- Task 1
For executing the first task, navigate to the script's directory and run:

python3 SN_NLP_HW4_task1.py

Ensure conlleval.py and model_hw4_task1.pth are in the directory.

- Task 2
For the second task, execute the following command in the terminal:

python3 SN_NLP_HW4_task2.py

The directory must contain conlleval.py, model_hw4_task2.pth, and glove.6B.100d.txt.

- Task 3
For executing the first task, navigate to the script's directory and run:

python3 SN_NLP_HW4_task3.py

Ensure conlleval.py and model_hw4_task3.pth are in the directory.

## Outputs

The evaluation results of the conlleval script will be displayed on the terminal, first for the validation then the test set. 
