HW2_generation_script.py - Execution instructions

Python version: 3.9.12

To successfully execute the python file, the following files need to be in the same directory as the script:
- train.json : the training data
- test.json : the test data
- hmm.json : the saved transition and emission dataset
- vocab.txt  : the vocabulary generated using training dataset

In order to generate the output json files for greedy and viterbi decoding, enter the following command in a terminal: 

python HW2_generation_script.py 

The script will automatically read the test set and generate the respective prediction files, 'greedy.json' and 'viterbi.json' respectively. 


