# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:14:47 2024

@author: dansc
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import string
import numpy as np

# =============================================================================
# LOAD DATA
# =============================================================================

# LOAD PREDS
rel_preds = pd.read_json('gemma_base_nit_WD_context_generations.jsonl', lines=True)['PREDICTION']
rel_preds[20] = 'My favorite type of icecream can be found in the buttercream'

rel_preds[0]

# LOAD ACTUAL
answers = pd.read_json('test.jsonl', lines=True)['answer']
answers[20] = ['hopscotch','buttercream']
answers[0]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def exact_match_f1(pred, answer_list):
    """
    Calculate the highest F1 score for a given prediction against a list of potential answers.

    Parameters:
    - pred (str): The prediction string to be evaluated.
    - answer_list (list of str): A list of answer strings against which the prediction is evaluated.
    
    Returns:
    - float: The highest F1 score achieved against any of the answers in the list.

    Each answer is compared to the prediction by splitting the strings into words and evaluating the overlap.
    The best F1 score is determined by comparing the harmonic mean of precision and recall for each answer.
    """
    pred_words = set(pred.lower().split())  # Convert the prediction into a set of words for faster operations
    best_f1 = 0  # Initialize best F1 score

    # Evaluate each answer in the list against the prediction
    for answer in answer_list:
        answer_words = set(answer.lower().split())  # Convert answer into a set of words
        TP = len(answer_words.intersection(pred_words))
        FP = len(pred_words.difference(answer_words))
        FN = len(answer_words.difference(pred_words))
        
        # Check if any true positives to avoid division by zero in precision and recall
        if TP == 0:
            f1 = 0
        else:
            prec = TP / (TP + FP) if TP + FP > 0 else 0
            rec = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * ((prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0

        # Update best F1 score if the current F1 score is higher
        if f1 > best_f1:
            best_f1 = f1
            
    return best_f1


def contains_metric(pred, answer_list):
    """
    Checks if any answer in the list is contained within the prediction after removing punctuation
    and converting to lowercase.

    Parameters:
    - pred (str): The prediction string to be evaluated.
    - answer_list (list of str): A list of answer strings against which the prediction is evaluated.

    Returns:
    - bool: True if any answer is contained within the prediction, False otherwise.
    """
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    normalized_pred = pred.lower().translate(translator)

    for answer in answer_list:
        # Normalize each answer
        normalized_answer = answer.lower().translate(translator)
        # Check if the normalized answer is contained within the normalized prediction
        if normalized_answer in normalized_pred:
            return 1

    return 0

# =============================================================================
# EXAMPLE USE
# =============================================================================
f1s = []
contains = []
print('hello')
for pred, ans_list in zip(rel_preds, answers):
     f1s.append(exact_match_f1(pred, ans_list))
     contains.append(contains_metric(pred, ans_list))

average_F1= sum(f1s) / len(f1s)
average_contains= sum(contains) / len(contains)

print("f1",average_F1)
print("contains",average_contains)
