import json
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm


EPSILON = 1e-10 # small constant used to avoid log(0) in calculations

# defining functions
def pseudo_word(word):
    """
    Convert a word into a pseudo-word based on its features.
    This method is based on the table from Bikel et al. (1999) 
    with additional categories.
    """
    if len(word) == 2 and word.isdigit():
        return "<twoDigitNum>"
    
    if len(word) == 4 and word.isdigit():
        return "<fourDigitNum>"
    
    if any(char.isdigit() for char in word) and any(char.isalpha() for char in word):
        return "<containsDigitAndAlpha>"
    
    if "-" in word and any(char.isdigit() for char in word):
        return "<containsDigitAndDash>"
    
    if "/" in word and any(char.isdigit() for char in word):
        return "<containsDigitAndSlash>"
    
    if "," in word and any(char.isdigit() for char in word):
        return "<containsDigitAndComma>"
    
    if "." in word and any(char.isdigit() for char in word):
        return "<containsDigitAndPeriod>"
    
    if word.isdigit() and not any([char in word for char in ["-", "/", ".", ","]]):
        return "<othernum>"
    
    if word.isupper():
        return "<allCaps>"
    
    if len(word) == 2 and word[1] == "." and word[0].isupper():
        return "<capPeriod>"
    
    if word[0].isupper():
        return "<initCap>"
    
    if word.islower():
        return "<lowercase>"

    return "<unk>"

# greedy supporting functions
def get_transition_probability(S1, S2, transition_dict):
    key = f"{S1},{S2}"
    return transition_dict.get(key, 1e-5) # 1e-8 in case nothing was found 
 
def get_emission_probability(S, X, emission_dict):
    key = f"{X},{S}"
    return emission_dict.get(key, emission_dict.get(f"<unk>,{S}",0.0))


# greedy decoding 
def greedy_decoding_state(S, word, transition_dict, emission_dict, state_set):
    """
    implements the greedy decoding algorithm for a given state (1 step)
    """
    max_p = float('-inf') # lowest number possible
    selected_state = None 

    for S1 in state_set:
        t = get_transition_probability(S, S1, transition_dict)
        e = get_emission_probability(S1, word, emission_dict)
        if t * e > max_p:
            max_p = t * e
            selected_state = S1
    return selected_state

def greedy_decoding(inputs, transition_dict, emission_dict, state_set):
    """
    implements the greedy decoding algorithm using greedy_decoding_state for a given sentence/list of words
    """
    state_sequence = []
    S = '<start>'  # Starting state
    for word in inputs:
        S = greedy_decoding_state(S, word, transition_dict, emission_dict, state_set)
        state_sequence.append(S)
    return state_sequence

# viterbi supporting functions
def generate_transition_matrix(transition_dict, ss):
    """
    Generates a transition matrix from transition_dict.
    Helps with faster calculation of the viterbi algorithm.
    """
    transition_matrix = [[transition_dict[ss[i] + ',' + ss[j]] for j in range(len(ss))] for i in range(len(ss))]
    return np.array(transition_matrix)

def generate_emmission_matrix(emission_dict, vocab, ss):
    """
    Generates an emission matrix from an emission_dict.
    Helps with faster calculation of the viterbi algorithm.
    """
    emmission_matrix = [[get_emission_probability(ss[i],vocab[j],emission_dict) for j in range(len(vocab))] for i in range(len(ss))] #emmission_dict[vocab[j] + ',' + ss[i]]
    # storing the results as a dataframe (easier access of data)
    matrix = pd.DataFrame(np.array(emmission_matrix).T, index=vocab)
    return matrix

def generate_initial_probabilities(transition_dict, ss):
    """
    Generates initial probabilities based on transition_dict.
    """
    initial = [transition_dict.get('<start>,' + state, 1e-6) for state in ss]
    return initial


def safe_log(x):
    """Compute the logarithm, but replace zeros or near-zeros with a small positive value first."""
    return np.log(np.where(np.abs(x) < EPSILON, EPSILON, x))

def viterbi_decoding_log(inputs, transition_matrix, emission_matrix, initial_probabilities, state_set):
    """
    Implements the Viterbi algorithm on the given inputs using log-space computations.
    """
    N = len(inputs)
    n_states = len(initial_probabilities)
    
    V = np.zeros((N, n_states)) - np.inf  # initialization of Viterbi matrix V to negative infinity in log-space
    
    # Initialize the first column of V based on initial probabilities and emission matrix
    first_word = inputs[0]
    if first_word in emission_matrix.index:
        V[0] = safe_log(initial_probabilities) + safe_log(emission_matrix.loc[first_word].values)
    else:
        V[0] = safe_log(initial_probabilities) + safe_log(emission_matrix.loc['<unk>'].values)
        
    # Fill in the rest of the Viterbi matrix
    for t in range(1, N):
        for s in range(n_states):
            word = inputs[t]
            if word in emission_matrix.index:
                V[t, s] = np.max(V[t-1] + safe_log(transition_matrix[:, s])) + safe_log(emission_matrix.loc[word, s])
            else:
                V[t, s] = np.max(V[t-1] + safe_log(transition_matrix[:, s])) + safe_log(emission_matrix.loc['<unk>', s])
    
    # Backtracking to find the most likely sequence
    back_tracking = [np.argmax(V[-1])]
    for i in range(N-2, -1, -1):
        back_tracking.append(np.argmax(V[i] + safe_log(transition_matrix[:, back_tracking[-1]])))
    
    # Reverse the backtracking result to get the final sequence
    result = back_tracking[::-1]
    decoded_sequence = [state_set[j] for j in result]
    
    return decoded_sequence

def make_sentences(data):
    """
    takes data as input and returns a set of sentences and their corresponding tags
    """
    return [[i['sentence'],i['labels']] for i in data]

# loading all data
with open('train.json') as f:
    train_data = json.load(f)

with open('test.json','r') as f:
    test_data = json.load(f)

with open('test.json','r') as f:
    test_data_copy = json.load(f)

# loading vocab
vocab = {}
with open('vocab.txt', 'r') as f:
    for line in f:
        word, _, freq = line.strip().split('\t')
        vocab[word] = int(freq)
        
# unique tags
unique_tags = set()  # Create an empty set to store unique tags
for item in train_data:
    unique_tags.update(item['labels'])

# replacing data with unknown words
for data_point in test_data:
    for i in range(len(data_point['sentence'])):
        if data_point['sentence'][i] not in vocab:
            data_point['sentence'][i] = pseudo_word(data_point['sentence'][i])

# loading HMM
with open('hmm.json','r') as f:
    hmm = json.load(f)

transition_probabilities = hmm['transition']
emission_probabilities = hmm['emission']

# greedy decoding on test set
greedy_test_preds = [{
    "index":test_data[x]['index'],
    "sentence":test_data_copy[x]['sentence'],
    'labels':greedy_decoding(test_data[x]['sentence'],transition_probabilities,emission_probabilities,unique_tags)} for x in range(len(test_data))]

extra_vocab_list = ["<twoDigitNum>","<fourDigitNum>","<containsDigitAndAlpha>",
                    "<containsDigitAndDash>","<containsDigitAndSlash>","<containsDigitAndComma>",
                     "<containsDigitAndPeriod>","<othernum>","<allCaps>","<capPeriod>","<initCap>","<lowercase>"] #"<hasHyphen>","<shortWord>", "<titleCase>", "<NESS_SUFFIX>", "<MENT_SUFFIX>","<ANTI_PREFIX>", "<SUPER_PREFIX>", "<hasSpecialSymbol>", "<negation>", "<EST_SUFFIX>" , "<IZE_SUFFIX>", "<repeatedChar>", "<EX_PREFIX>", "<hasColon>", "<hasSemiColon>"]
EM = generate_emmission_matrix(emission_probabilities,list(vocab.keys())+extra_vocab_list,list(unique_tags))

TM = generate_transition_matrix(transition_probabilities,list(unique_tags))
initial = generate_initial_probabilities(transition_probabilities,list(unique_tags))

viterbi_test_preds = [{
    "index":test_data[x]['index'],
    "sentence":test_data_copy[x]['sentence'],
    'labels':viterbi_decoding_log(test_data[x]['sentence'],TM,EM,initial,list(unique_tags))} for x in range(len(test_data))]


with open('greedy.json','w') as f:
    json.dump(greedy_test_preds,f)

with open('viterbi.json','w') as f:
    json.dump(viterbi_test_preds,f)