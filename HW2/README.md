# CSCI 544: Assignment 2 Report

## Grade: 100/100

## Overview
This report outlines the implementation details of Assignment 2 for CSCI 544, which is divided into four main tasks: Vocabulary Creation, Model Learning, Greedy Decoding, and Viterbi Decoding.

## 1. Vocabulary Creation
- **Threshold Determination:** After several test runs, a threshold of 2 was selected for unknown word occurrences, balancing vocabulary size and the number of unknown words.
- **Statistics:**
  - Threshold set: 2
  - Vocabulary Size: 16920
  - Unknown Word Occurrences: 32537
- **Pseudo-words Implementation:** Implemented the concept of pseudo-words for better handling of unknown words.

## 2. Model Learning
- **Transition and Emission Dictionaries:** Developed functions to calculate state transition and state emission occurrences, facilitating the creation of transition and emission dictionaries.
- **Parameters:**
  - Number of transition parameters: 2070
  - Number of emission parameters: 761492

## 3. Greedy Decoding
- **Algorithm Implementation:** Implemented a greedy decoding algorithm using the transition and emission dictionaries.
- **Accuracy:** Achieved an accuracy of 93.94% on the development set.

## 4. Viterbi Decoding
- **Optimization:** Initially implemented using nested loops, then optimized with a matrix multiplication approach.
- **Accuracy Improvement:** Applied log-space computations and Laplace smoothing to improve accuracy.
- **Final Accuracy:** Achieved an accuracy of 94.98% on the development set.

---

*Note: Detailed code and analysis are provided in the attached Jupyter notebooks.*
